"""
Custom Ollama LLM adapter for Google ADK.

Bridges the ADK BaseLlm interface with Ollama's Python client so that
any Ollama model (including VLMs like qwen3-vl) works transparently
inside ADK agents — without requiring LiteLLM.

Registration:  import agent.ollama_llm   →   LLMRegistry sees "ollama/.*"
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, AsyncGenerator

import ollama
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types
from typing_extensions import override

from config import settings

log = logging.getLogger(__name__)


# helpers 

def _schema_to_json(schema: types.Schema | None) -> dict:
    """Recursively convert google.genai Schema to plain JSON-Schema dict."""
    if schema is None:
        return {"type": "object", "properties": {}}

    result: dict[str, Any] = {}

    if schema.type is not None:
        result["type"] = schema.type.value.lower()  # e.g. "string", "object"
    if schema.description:
        result["description"] = schema.description
    if schema.enum:
        result["enum"] = schema.enum
    if schema.properties:
        result["properties"] = {k: _schema_to_json(v) for k, v in schema.properties.items()}
    if schema.required:
        result["required"] = schema.required
    if schema.items:
        result["items"] = _schema_to_json(schema.items)

    return result


def _extract_system_text(si: Any) -> str:
    """Pull plain text from whatever ADK puts in system_instruction."""
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    if isinstance(si, types.Part):
        return si.text or ""
    if isinstance(si, types.Content):
        return " ".join(p.text for p in (si.parts or []) if p.text)
    if isinstance(si, list):
        return " ".join(_extract_system_text(item) for item in si)
    return str(si)


# main class─

class OllamaLlm(BaseLlm):
    """
    ADK-compatible wrapper around Ollama.

    Model name format:  "ollama/<ollama-model-tag>"
    Example:            "ollama/qwen3-vl:8b"
    """

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"ollama/.*"]

    # private helpers 

    def _model_tag(self) -> str:
        """Strip 'ollama/' prefix to get the bare Ollama model tag."""
        return self.model.removeprefix("ollama/")

    def _build_tools(self, llm_request: LlmRequest) -> list[dict] | None:
        """Convert ADK FunctionDeclarations → Ollama tool dicts."""
        if not llm_request.config or not llm_request.config.tools:
            return None

        ollama_tools: list[dict] = []
        for tool in llm_request.config.tools:
            if not hasattr(tool, "function_declarations"):
                continue
            for fd in tool.function_declarations or []:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": fd.name,
                        "description": fd.description or "",
                        "parameters": _schema_to_json(fd.parameters),
                    },
                })
        return ollama_tools or None

    def _build_messages(self, llm_request: LlmRequest) -> list[dict]:
        """Convert ADK Content list → Ollama message list."""
        messages: list[dict] = []

        # System instruction
        if llm_request.config:
            sys_text = _extract_system_text(llm_request.config.system_instruction)
            if sys_text:
                messages.append({"role": "system", "content": sys_text})

        for content in llm_request.contents or []:
            role = "assistant" if content.role == "model" else (content.role or "user")

            text_parts: list[str] = []
            fn_calls: list[types.FunctionCall] = []
            fn_responses: list[types.FunctionResponse] = []
            images: list[bytes] = []

            for part in content.parts or []:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    fn_calls.append(part.function_call)
                elif part.function_response:
                    fn_responses.append(part.function_response)
                elif part.inline_data and part.inline_data.data:
                    images.append(part.inline_data.data)

            if fn_calls:
                # Model wanted to call tools
                messages.append({
                    "role": "assistant",
                    "content": " ".join(text_parts),
                    "tool_calls": [
                        {
                            "function": {
                                "name": fc.name,
                                "arguments": fc.args or {},
                            }
                        }
                        for fc in fn_calls
                    ],
                })
            elif fn_responses:
                # Tool results (one Ollama message per result)
                for fr in fn_responses:
                    resp = fr.response or {}
                    content_str = json.dumps(resp) if isinstance(resp, dict) else str(resp)
                    messages.append({"role": "tool", "content": content_str})
            else:
                msg: dict[str, Any] = {
                    "role": role,
                    "content": " ".join(text_parts),
                }
                if images:
                    # VLM: pass images as base64 strings
                    msg["images"] = [
                        base64.b64encode(img).decode() if isinstance(img, bytes) else img
                        for img in images
                    ]
                messages.append(msg)

        return messages

    def _to_llm_response(self, response: ollama.ChatResponse) -> LlmResponse:
        """Convert an Ollama ChatResponse to an ADK LlmResponse."""
        msg = response.message
        parts: list[types.Part] = []

        if msg.content:
            parts.append(types.Part(text=msg.content))

        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc.function.name,
                        args=args,
                    )
                )
            )

        content = types.Content(role="model", parts=parts) if parts else None
        return LlmResponse(content=content, partial=False, turn_complete=True)

    # core interface ─

    @override
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        messages = self._build_messages(llm_request)
        tools = self._build_tools(llm_request)
        model_tag = self._model_tag()

        client = ollama.AsyncClient(host=settings.ollama_base_url)

        # Ollama tool calling only works reliably in non-streaming mode
        has_tools = bool(tools)

        if stream and not has_tools:
            # Stream text tokens
            full_text = ""
            async for chunk in await client.chat(
                model=model_tag,
                messages=messages,
                stream=True,
            ):
                delta = chunk.message.content or ""
                full_text += delta
                if delta:
                    yield LlmResponse(
                        content=types.Content(
                            role="model", parts=[types.Part(text=delta)]
                        ),
                        partial=True,
                    )
            # Final consolidated response
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=full_text)]
                ),
                partial=False,
                turn_complete=True,
            )
        else:
            # Non-streaming (required for tool calls)
            kwargs: dict[str, Any] = {
                "model": model_tag,
                "messages": messages,
                "stream": False,
            }
            if tools:
                kwargs["tools"] = tools

            log.debug("[OllamaLlm] → %s  messages=%d  tools=%s",
                      model_tag, len(messages), [t["function"]["name"] for t in (tools or [])])

            response = await client.chat(**kwargs)

            log.debug("[OllamaLlm] ← content=%r  tool_calls=%r",
                      response.message.content, response.message.tool_calls)

            yield self._to_llm_response(response)


# Register so that ADK resolves "ollama/..." → OllamaLlm
LLMRegistry.register(OllamaLlm)
