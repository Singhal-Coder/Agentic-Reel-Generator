# agents/extractor.py

from ..models.specs import CanonicalSpec, FieldWithExplicit, OverlayElementType
from typing import Dict, Any, Optional, List, Literal
from pydantic import Field

from .llm_utils import call_llm_with_json_output
from ..providers.loader import providers
from ..schemas.user_params import UserParams
from ..utils.logging import logger

class ExtractedSpec(UserParams):
    title: Optional[str] = Field(None, description="The topic or title of the reel based on the user's prompt.")
    mood: Optional[str] = Field(None, description="The desired mood (e.g., 'a bit sad but hopeful').")
    pacing: Optional[str] = Field(None, description="The desired pacing (e.g., 'starts slow, then gets fast').")
    overlay_elements: Optional[List[OverlayElementType]] = Field(None, description="A list of generic overlay categories mentioned.")
    denies: Optional[List[str]] = Field(
        None, 
        description="A list of topics, elements, or specific visuals to explicitly avoid (e.g., 'battles', 'showing brand names')."
    )
    extra_instructions: Optional[List[str]] = Field(
        None, 
        description="A list of specific, non-narrative stylistic notes or constraints that ARE NOT part of the visual story (e.g., 'use a vintage film look', 'ensure all text is in the upper third of the screen'). Do not contain shot descriptions, scene transitions, story beats, or narrative elements."
    )

EXTRACTOR_SYSTEM_PROMPT = """
You are a precise data extraction agent. Your sole purpose is to extract specific user requirements from a given text prompt and format them into a JSON object that conforms to the provided schema.

**Your Rules:**
1.  **Map Constrained Fields with High Confidence:** For fields with fixed options (`aspect_ratio`, `voice_style`, `music_style`), you must follow this logic:
    a. First, check if the user's request is an **exact match** for one of the available options. If so, use it.
    b. If not, check if it is a **highly similar match** (e.g., a user asks for 'pop music', you can confidently map it to 'upbeat_electronic'). If so, use the mapped value.
    c. If the user's request is not highly similar to any available option (e.g., 'Gregorian chants'), you **MUST leave the field empty**. Do not guess a poor match.
2.  **Strict Extraction for Other Fields:** For all other fields, only extract information that the user has explicitly stated. Do NOT infer or add creative details.
3.  **Omit Missing Fields:** If the user does not mention a specific field, you MUST omit it from the JSON output.
4.  **Match the Schema:** The output must be a valid JSON object that strictly follows the provided Pydantic model schema.

**Example:**
* **User Request:** "Make a fast-paced 10s reel about ancient Rome, don't show any battles. Make it look epic, use warm colors and add a #history hashtag"
* **Your JSON Output:**
    ```json
    {
      "title": "ancient Rome",
      "duration_sec": 10,
      "pacing": "fast-paced",
      "overlay_elements": ["hashtags"],
      "denies": ["battles"],
      "extra_instructions": ["make it look epic", "use warm colors", "add a #history hashtag"]
    }
    ```
"""

def extract_instructions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses a constrained LLM to extract explicit user requests from the text prompt,
    merges them with structured parameters, and builds the initial CanonicalSpec.
    """
    user_prompt = state.get("prompt", "")
    params = state.get("params", {})

    extracted_data = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_model=ExtractedSpec
    )

    combined_data = {**extracted_data, **params}

    spec = CanonicalSpec()
    for key, value in combined_data.items():
        if hasattr(spec, key) and value is not None:
            if isinstance(value, list) and not value:
                continue
            field = FieldWithExplicit(value=value, explicit=True)
            setattr(spec, key, field)

    logger.info("--- Instruction Extractor: Populated initial spec from LLM ---")
    logger.info(spec.model_dump_json(indent=2))

    return {"spec": spec}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(extract_instructions(state))
    save_app_state(state)