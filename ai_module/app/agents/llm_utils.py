# agents/llm_utils.py

import json
import re
import ast
from typing import Dict, Any, Type, List, Optional
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser


def _strip_code_fence(text: str) -> str:
    """Remove surrounding ```json ... ``` or ``` ... ``` and trim whitespace."""
    if not text:
        return text
    # remove leading/trailing triple-backticks (optionally with json)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_text_from_response(resp: Any) -> str:
    """Try common attributes (.content, .text) or fallback to str()."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    # common attribute names
    for attr in ("content", "text", "message", "response"):
        val = getattr(resp, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    return str(resp)


def _try_load_jsonish(text: str) -> Optional[dict]:
    """
    Try json.loads, then ast.literal_eval. Return dict on success, else None.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def call_llm_with_json_output(
    llm_provider: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    output_model: Type[BaseModel],
    raise_on_error: bool = False,
    max_fallback_attempts: int = 1,
) -> Dict[str, Any]:
    """
    Robustly call an LLM and return a dict matching `output_model`.

    Behavior:
     - First try provider.with_structured_output(...) (if supported).
     - If that doesn't produce a dict/model, fallback to calling the model
       normally and parsing text with PydanticOutputParser + JSON/ast fallbacks.
     - By default it prints errors and returns {} on failure; set raise_on_error=True
       to raise exceptions instead.

    Args:
        llm_provider: BaseChatModel instance.
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        output_model: Pydantic model class describing expected output.
        raise_on_error: If True, raise on parse failure. Default False.
        max_fallback_attempts: number of times to fallback-call the model (if needed).

    Returns:
        dict parsed from model output (or {} on failure when raise_on_error=False).
    """
    parser = PydanticOutputParser(pydantic_object=output_model)

    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessage(content=user_prompt),
    ]
    last_error = None
    for attempt in range(max_fallback_attempts):
        try:
            raw_resp = llm_provider.invoke(messages)
        except Exception as e:
            last_error = e
            print(f"llm_provider.invoke failed on attempt {attempt+1}: {e}")
            continue

        text = _extract_text_from_response(raw_resp)
        text_clean = _strip_code_fence(text)

        try:
            parsed = parser.parse(text_clean)
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump()
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            last_error = e
            pass

        try:
            obj = _try_load_jsonish(text_clean)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_error = e

        print(f"Attempt {attempt+1} parse failed. Raw response (first 500 chars):\n{text_clean[:500]}")

    err_msg = f"Could not parse LLM response into dict for model {output_model}. Last error: {last_error}"
    print("Error calling LLM with structured output:", err_msg)
    if raise_on_error:
        raise ValueError(err_msg)
    return {}
