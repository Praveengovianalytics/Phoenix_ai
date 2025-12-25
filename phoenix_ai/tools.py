from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


class ChatAdapter(Protocol):
    def create_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """
        Returns a tuple of (final_content, tool_calls).
        If final_content is not None and tool_calls is empty, the conversation is done.
        If tool_calls are present, they should be executed, appended to messages, and the loop continues.
        """


class OpenAIStyleAdapter:
    """
    Adapter for OpenAI-compatible providers (OpenAI, Azure OpenAI, Databricks, Ollama OpenAI API).
    Expects a client with .chat.completions.create(...).
    """

    def __init__(self, openai_compatible_client: Any, model: str):
        self._client = openai_compatible_client
        self._model = model

    def create_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        tools_payload = None
        if tools:
            tools_payload = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools_payload,
            tool_choice=tool_choice or "auto",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        message = response.choices[0].message
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls: List[ToolCall] = []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )
        if tool_calls:
            return None, tool_calls
        return message.content or "", []


class JsonFunctionAdapter:
    """
    Provider-agnostic adapter: instructs the model to emit a JSON tool call plan in its content.
    This works even if the provider has no native tool-calling. The JSON format is:
    {"tool_calls":[{"name":"<tool>","arguments":{...}}]}
    If no tool is needed, return {"final_answer":"..."}.
    """

    def __init__(self, send_chat: Callable[[List[Dict[str, Any]]], str]):
        self._send_chat = send_chat

    def create_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        tool_descriptions = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in (tools or [])
        ]

        instruction = (
            "You can call tools. Decide next step and reply with a JSON object only.\n"
            "If you need a tool: {\"tool_calls\": [{\"name\": \"...\", \"arguments\": {...}}]}\n"
            "If you can answer directly: {\"final_answer\": \"...\"}.\n"
            f"Available tools: {json.dumps(tool_descriptions)}"
        )

        planned_messages = messages + [{"role": "system", "content": instruction}]
        content = self._send_chat(planned_messages)

        try:
            parsed = json.loads(content)
        except Exception:
            # Fallback: treat as final answer
            return content, []

        if isinstance(parsed, dict) and "tool_calls" in parsed:
            tool_calls = [
                ToolCall(id=str(i), name=tc.get("name", ""), arguments=tc.get("arguments", {}))
                for i, tc in enumerate(parsed.get("tool_calls", []))
            ]
            return None, tool_calls

        if isinstance(parsed, dict) and "final_answer" in parsed:
            return str(parsed.get("final_answer", "")), []

        # Unknown structure; return as final content
        return content, []


def run_agent_loop(
    adapter: ChatAdapter,
    messages: List[Dict[str, Any]],
    tools: List[Tool],
    max_steps: int = 8,
    on_tool_result: Optional[Callable[[str, Any], None]] = None,
) -> str:
    """
    Generic tool execution loop. Adapter can be OpenAI-style or JSON-style. Tools are provider-agnostic.
    """
    name_to_tool: Dict[str, Tool] = {t.name: t for t in tools}
    steps = 0
    while True:
        if steps >= max_steps:
            return "Tool execution exceeded max steps."

        final_content, tool_calls = adapter.create_response(messages, tools)
        if final_content is not None and not tool_calls:
            return final_content

        for tc in tool_calls:
            tool = name_to_tool.get(tc.name)
            if tool is None:
                result: Any = {"error": f"Unknown tool: {tc.name}"}
            else:
                try:
                    result = tool.function(**tc.arguments)
                except Exception as tool_error:
                    result = {"error": str(tool_error)}

            if on_tool_result:
                try:
                    on_tool_result(tc.name, result)
                except Exception:
                    pass

            # Normalize tool result to string
            if not isinstance(result, str):
                try:
                    content = json.dumps(result)
                except Exception:
                    content = str(result)
            else:
                content = result

            # Append tool result message in a provider-agnostic way.
            # For OpenAI-style, adapter expects a tool role message with tool_call_id; since adapters own the send,
            # we keep it generic: add a synthetic message the next call can use as context.
            messages.append(
                {
                    "role": "tool",
                    "name": tc.name,
                    "content": content,
                    "tool_call_id": tc.id,
                }
            )

        steps += 1


