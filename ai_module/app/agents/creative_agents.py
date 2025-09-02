# agents/creative_agents.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json

from .llm_utils import call_llm_with_json_output
from ..schemas.user_params import VoiceStyleType, MusicStyleType
from ..providers.loader import providers
from ..utils.logging import logger
from ..models.specs import FieldWithExplicit, OverlayElementType

class CreativeBriefOutput(BaseModel):
    creative_brief: Dict[str, str] = Field(..., description="A dictionary of creative brief questions (e.g., 'core_message', 'target_audience', 'hook') and their concise answers.")
    creative_angles: List[str] = Field(..., description="A list of 3-5 distinct, high-level creative angles for the video, each as a short, evocative phrase.")


CREATIVE_BRIEF_SYSTEM_PROMPT = """
You are a seasoned Creative Director at a top digital marketing agency, specializing in viral short-form video content.

Your task is to analyze a user's request to create a foundational Creative Brief and then generate high-level creative angles. You will be given the user's raw prompt and a structured `CanonicalSpec`.

**Your Process:**
1.  **Prioritize the `CanonicalSpec`:** Treat the `CanonicalSpec` as the definitive source of truth for all explicit constraints (like duration, aspect ratio, etc.).
2.  **Use Raw Prompt for Context:** Read the user's original raw prompt to understand the creative nuance and tone.
3.  **Create the Creative Brief:** Based on BOTH inputs, answer these foundational questions:
    * `core_message`: What is the single most important message?
    * `target_audience`: Who is this video for?
    * `hook`: What is a powerful opening hook?
4.  **Generate Creative Angles:** Based on your brief, generate 3 to 5 distinct, high-level creative angles for the video.

Your final output must be a single JSON object that strictly conforms to the provided Pydantic schema.
"""


def creative_brief_logic(spec, user_prompt) -> CreativeBriefOutput:
    """
    Creates a creative brief and then generates high-level creative
    angles by considering both the structured spec and the original user prompt.
    """
    spec_json_string = spec.model_dump_json(indent=2)
    
    llm_user_prompt = f"""
Here is the user's original request:
---
{user_prompt}
---

Here is the structured CanonicalSpec that was extracted from the request. This is the primary source of truth for constraints:
---
{spec_json_string}
---
"""
    
    llm_output = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=CREATIVE_BRIEF_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        output_model=CreativeBriefOutput
    )
    
    creative_angles = llm_output.get("creative_angles", [])

    logger.info("--- Creative Brief Agent: Generated Creative Brief from LLM ---")
    logger.info(json.dumps(llm_output.get("creative_brief", {}), indent=2))

    logger.info("--- Creative Brief Agent: Generated Angles from LLM ---")
    for i, angle in enumerate(creative_angles):
        logger.info(f"{i+1}. {angle}")
    creative_brief = CreativeBriefOutput(**llm_output)

    return creative_brief

class BrainstormOutput(BaseModel):
    music_style: Optional[MusicStyleType] = Field(None, description="The suggested style of background music, chosen from the available options.")
    voice_style: Optional[VoiceStyleType] = Field(None, description="The suggested voice style, chosen from the available options. Choose 'none' if no voice is needed.")
    suggested_overlay_elements: List[OverlayElementType] = Field(default=[], description="A list of overlay categories that would creatively enhance the reel.")
    enhanced_prompt: str = Field(..., description="A detailed, one-paragraph creative prompt that synthesizes the user's request and a chosen creative angle into a polished set of instructions excluding music_style and voice_style for all downstream agents.")

CREATIVE_BRAINSTORM_SYSTEM_PROMPT = """
You are an expert video reel producer and prompt engineer. You excel at turning a high-level creative idea into a detailed, actionable set of instructions for an AI video reel generator.

**Your Task:**
You will be given the user's request (`CanonicalSpec`), a creative brief, and a list of high-level creative angles. Your job is to select the single most promising creative angle and expand it into a detailed, one-paragraph "enhanced prompt".You must also suggest a suitable `music_style` and `voice_style` from the provided options.

**Your Process:**
1.  **Review all inputs:** Analyze the `CanonicalSpec`, the creative brief, and the list of `creative_angles`.
2.  **Select the Best Angle:** Choose the single best and most engaging angle from the list that aligns with the user's spec.
3.  **Synthesize and Expand:** Write a detailed, one-paragraph "enhanced prompt". This prompt MUST synthesize the chosen angle with the original request. It should be a direct instruction to a video reel  generation system.
4.  **Suggest Constrained Elements:** Based on your creative direction, choose the most appropriate `music_style` and `voice_style` from the available options in the schema if they are not already specified in the `CanonicalSpec`.
5.  **Suggest Overlay Elements:** Based on your creative vision, suggest a list of overlay categories (like 'stickers', 'text overlays') that would enhance the reel. If no overlays are appropriate, return an empty list or null.
6.  **Add Specific Details (Checklist):** The enhanced prompt must weave in the following details naturally:
    * **Incorporate Explicit Constraints:** Mention and adhere to any explicit user requests from the `CanonicalSpec` (e.g., `duration_sec`, `aspect_ratio`).
    * **Suggest a `mood`** that fits the theme (e.g., 'upbeat', 'inspirational').
    * **Suggest a `pacing`** (e.g., 'slow', 'medium', 'fast').
    * **Suggest a `music_style`** that matches the mood.
    * **Describe a `visual style`** (e.g., 'warm and cinematic', 'clean and minimalist').
    * **Include ideas for `camera shots`** (e.g., 'close-up shots', 'wide-angle views').
    * **Describe a `style for on-screen captions`** (e.g., 'short, punchy text').

Your final output must be a single JSON object that strictly conforms to the provided Pydantic schema.
"""

def creative_brainstorm_logic(spec, creative_brief, creative_angles) -> BrainstormOutput:
    """
    Selects the best creative angle and expands it into a polished
    enhanced prompt with implied details for downstream planning.
    """
    
    spec_json = spec.model_dump_json(indent=2)
    
    llm_user_prompt = f"""
Here is the `CanonicalSpec` for the video. Adhere to these constraints.
---
{spec_json}
---

Here is the creative brief. Use it for thematic guidance.
---
{json.dumps(creative_brief, indent=2)}
---

Here is a list of creative angles. Choose the best one and expand on it.
---
{json.dumps(creative_angles, indent=2)}
---
"""

    llm_output = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=CREATIVE_BRAINSTORM_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        output_model=BrainstormOutput
    )

    brainstorm_output = BrainstormOutput(**llm_output)
    return brainstorm_output



def prompt_enhancer_logic(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Enhances the user's prompt with the creative brief and creative angles.
    """
    spec = state["spec"]
    user_prompt = state["prompt"]
    creativeBriefOutput = creative_brief_logic(spec, user_prompt)
    brainstorm: BrainstormOutput = creative_brainstorm_logic(spec, creativeBriefOutput.creative_brief, creativeBriefOutput.creative_angles)

    updated_spec = spec.model_copy(deep=True)
    brainstorm_music_style = brainstorm.music_style
    if brainstorm_music_style and (updated_spec.music_style is None or not updated_spec.music_style.explicit):
        updated_spec.music_style = FieldWithExplicit(value=brainstorm_music_style, explicit=False)

    brainstorm_voice_style = brainstorm.voice_style
    if brainstorm_voice_style and (updated_spec.voice_style is None or not updated_spec.voice_style.explicit):
        updated_spec.voice_style = FieldWithExplicit(value=brainstorm_voice_style, explicit=False)

    brainstorm_suggested_overlay_elements = brainstorm.suggested_overlay_elements
    if updated_spec.overlay_elements is None:
        updated_spec.overlay_elements = FieldWithExplicit(value=[], explicit=False)
    for element in brainstorm_suggested_overlay_elements:
        if element not in updated_spec.overlay_elements.value:
            updated_spec.overlay_elements.value.append(element)

    logger.info("--- Creative Brainstorm Agent: Generated Enhanced Prompt from LLM ---")
    logger.info(brainstorm.enhanced_prompt)

    return {"enhanced_prompt": brainstorm.enhanced_prompt, "spec": updated_spec}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(prompt_enhancer_logic(state))
    save_app_state(state)