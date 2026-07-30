from __future__ import annotations

import re
from collections.abc import Callable, Mapping, MutableSequence
from typing import Any

from ollama import chat, show, web_fetch, web_search

MODEL = "hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0"
MAX_TOOL_ROUNDS = 8
MAX_TOOL_RESULT_CHARS = 8_000

AVAILABLE_TOOLS: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_TERMINATION_PATTERNS = (
    re.compile(
        r"(?:bye|bye bye|goodbye|good bye|farewell)"
        r"(?: (?:for now|now|thanks|thank you))?"
    ),
    re.compile(
        r"(?:please )?"
        r"(?:exit|quit|terminate|termination|stop|end|close|cancel|shutdown|disconnect|leave)"
        r"(?: (?:the )?(?:chat|conversation|session|program|assistant))?"
        r"(?: (?:please|now|thanks|thank you))?"
    ),
    re.compile(r"(?:i am|im) (?:done|finished)(?: (?:here|now|thanks|thank you))?"),
    re.compile(r"(?:thats|that is) all(?: (?:for now|thanks|thank you))?"),
    re.compile(r"see (?:you|ya)(?: later)?"),
)


def _normalize_command(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized.startswith("/"):
        normalized = normalized[1:].lstrip()
    normalized = normalized.replace("'", "").replace("\N{RIGHT SINGLE QUOTATION MARK}", "")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def is_termination_command(value: str) -> bool:
    """Return True only when the complete input expresses termination intent."""

    normalized = _normalize_command(value)
    return bool(normalized) and any(
        pattern.fullmatch(normalized) for pattern in _TERMINATION_PATTERNS
    )


def _visible_content(value: str | None) -> str:
    """Remove fallback think tags without ever exposing the thinking field."""

    return _THINK_BLOCK.sub("", value or "").strip()


def _assistant_history_message(message: Any, content: str) -> dict[str, Any]:
    """Create protocol history without retaining hidden model reasoning."""

    if hasattr(message, "model_dump"):
        history_message = message.model_dump(exclude={"thinking"}, exclude_none=True)
        history_message["content"] = content
        return history_message

    history_message: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        history_message["tool_calls"] = tool_calls
    return history_message


def _tool_result(tool_call: Any, tool_functions: Mapping[str, Callable[..., Any]]) -> str:
    tool_name = tool_call.function.name
    function_to_call = tool_functions.get(tool_name)
    if function_to_call is None:
        return f"Tool {tool_name} is not available."

    try:
        result = function_to_call(**tool_call.function.arguments)
    except Exception as exc:
        # Do not expose URLs, tokens, response bodies, or other sensitive details.
        return f"Tool {tool_name} failed with {type(exc).__name__}."

    return str(result)[:MAX_TOOL_RESULT_CHARS]


def tools_for_model(
    *, show_function: Callable[[str], Any] = show
) -> Mapping[str, Callable[..., Any]]:
    """Enable tools only when Ollama reports that the model supports them."""

    try:
        capabilities = show_function(MODEL).capabilities or []
    except Exception:
        # Chat remains useful when model metadata is temporarily unavailable.
        return {}
    return AVAILABLE_TOOLS if "tools" in capabilities else {}


def complete_assistant_turn(
    messages: MutableSequence[Any],
    *,
    chat_function: Callable[..., Any] = chat,
    tool_functions: Mapping[str, Callable[..., Any]] | None = None,
) -> str:
    """Complete one user turn, including any bounded Ollama tool-call rounds."""

    selected_tools = tool_functions or {}
    for tool_round in range(MAX_TOOL_ROUNDS + 1):
        chat_arguments: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "think": False,
        }
        if selected_tools:
            chat_arguments["tools"] = list(selected_tools.values())

        response = chat_function(**chat_arguments)
        response_message = response.message
        content = _visible_content(getattr(response_message, "content", None))
        messages.append(_assistant_history_message(response_message, content))

        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            return content
        if tool_round == MAX_TOOL_ROUNDS:
            raise RuntimeError("Ollama exceeded the allowed tool-call rounds")

        for tool_call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "content": _tool_result(tool_call, selected_tools),
                    "tool_name": tool_call.function.name,
                }
            )

    raise RuntimeError("Ollama did not complete the assistant turn")


def run_chat(
    *,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], Any] = print,
    chat_function: Callable[..., Any] = chat,
    tool_functions: Mapping[str, Callable[..., Any]] | None = None,
) -> None:
    """Run an in-process, multi-turn terminal chat session."""

    messages: list[Any] = []
    selected_tools = tools_for_model() if tool_functions is None else tool_functions
    output_function("Medical Assistant ready. Type /bye to end the chat.")

    while True:
        try:
            user_input = input_function("You: ")
        except (EOFError, KeyboardInterrupt):
            output_function("\nGoodbye.")
            return

        if not user_input.strip():
            continue
        if is_termination_command(user_input):
            output_function("Goodbye.")
            return

        turn_start = len(messages)
        messages.append({"role": "user", "content": user_input.strip()})

        try:
            content = complete_assistant_turn(
                messages,
                chat_function=chat_function,
                tool_functions=selected_tools,
            )
        except Exception as exc:
            del messages[turn_start:]
            output_function(
                f"Assistant error ({type(exc).__name__}). Please try again."
            )
            continue

        output_function(f"Assistant: {content or 'No response was returned.'}")


if __name__ == "__main__":
    run_chat()
