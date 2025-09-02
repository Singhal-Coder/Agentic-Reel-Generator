# agents/blueprint_agent.py

from typing import Dict, Any

from .llm_utils import call_llm_with_json_output
from ..providers.loader import providers
from ..models.specs import CanonicalSpec, FieldWithExplicit
from ..utils.logging import logger
from .extractor import ExtractedSpec

BLUEPRINT_SYSTEM_PROMPT = """
You are a highly logical and meticulous Production Assistant. Your job is to finalize a technical specification sheet (`CanonicalSpec`) by reconciling a creative brief (`enhanced_prompt`) with the original client demands.

**Your Task:**
You will be given an 'enhanced_prompt' and the current `CanonicalSpec`. Your task is to finalize the spec by inferring any missing creative parameters and ensuring all instructions are coherent and non-contradictory.

**Your Constitutional Rules (MUST be followed):**
1.  **The Hierarchy of Truth:** The `CanonicalSpec` is the source of truth. Any field marked with `"explicit": true` is a non-negotiable client demand. The `enhanced_prompt` is a creative guide that MUST operate within the boundaries of these explicit demands. If the `enhanced_prompt` suggests something that contradicts an explicit field, you MUST prioritize the explicit field and ignore the contradictory suggestion.
2.  **Infer Missing Creative Parameters:** Infer descriptive values for `title`, `mood` and `pacing`.
3.  **Synthesize and De-conflict Lists:** For list-based fields (`overlay_elements`, `denies`, `extra_instructions`), your task is to synthesize a final, clean list. To do this:
    a. Start with the existing items in the spec's list.
    b. Add any NEW, relevant items suggested by the `enhanced_prompt`.
    c. **Crucially, you must review the final synthesized list and REMOVE any items that contradict the explicit demands in the `CanonicalSpec`.** For example, if `denies` explicitly contains ["battles"], you must not add an instruction like "show an epic fight scene" to `extra_instructions`.
4.  **Final Output:** Your JSON output should ONLY contain the final, synthesized values for the fields you have updated. Do not return fields that have not changed.
"""

# --- The main agent logic ---
def blueprint_logic(state: Dict[str, Any]) -> Dict[str, CanonicalSpec]:
    """
    Parses the enhanced_prompt to populate the final CanonicalSpec,
    respecting any fields that were explicitly set by the user or previous agents.
    """
    spec = state["spec"]
    enhanced_prompt = state["enhanced_prompt"]

    spec_json = spec.model_dump_json(indent=2)
    
    llm_user_prompt = f"""
Here is the current `CanonicalSpec`. Your job is to fill in any remaining empty fields by parsing the `enhanced_prompt` below.
---
{spec_json}
---

Here is the creative `enhanced_prompt`.
---
{enhanced_prompt}
---
"""

    inferred_values = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=BLUEPRINT_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        output_model=ExtractedSpec
    )

    updated_spec = spec.model_copy(deep=True)

    for key, value in inferred_values.items():
        if not hasattr(updated_spec, key) or value is None:
            continue

        existing_field = getattr(updated_spec, key)
        
        if isinstance(value, list) and value:
            if existing_field is None:
                setattr(updated_spec, key, FieldWithExplicit(value=value, explicit=False))
            elif existing_field.value and isinstance(existing_field.value, list):
                existing_items = set(existing_field.value)
                for item in value:
                    if item not in existing_items:
                        existing_field.value.append(item)
        
        elif not isinstance(value, list) and (existing_field is None or not existing_field.explicit):
            setattr(updated_spec, key, FieldWithExplicit(value=value, explicit=False))

    logger.info("--- Blueprint Agent: Finalized CanonicalSpec from LLM ---")
    logger.info(updated_spec.model_dump_json(indent=2))

    return {"spec": updated_spec}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(blueprint_logic(state))
    save_app_state(state)