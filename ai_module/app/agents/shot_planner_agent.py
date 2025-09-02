# agents/shot_planner_agent.py

from typing import Dict, Any, List, get_args
from pydantic import BaseModel, Field
import json

from .llm_utils import call_llm_with_json_output
from ..providers.loader import providers
from ..models.specs import CanonicalSpec, ShotEntry
from ..utils.logging import logger

# --- 1. Define the Pydantic Model for the Agent's Output ---
# We already have ShotEntry defined, so we just need a wrapper for the list.
class ShotListOutput(BaseModel):
    shot_list: List[ShotEntry] = Field(..., description="The final, timestamped list of shots for the video, with descriptive tags for each shot.")

# --- 2. Create the System Prompt ---
SHOT_PLANNER_SYSTEM_PROMPT = """
You are a meticulous video editor and shot planner. Your expertise lies in translating a script into a visually compelling sequence of shots.

**Your Task:**
Create a detailed `ShotList` from a given script and a `CanonicalSpec`. You will determine the timing for each shot based on the script's timing, and for each shot, you will generate a complete visual description using all available fields.

**Your Process and Rules:**
1.  **Follow the Script's Timing:** The script is your primary guide. For each line in the script, you must create a corresponding `ShotEntry`. The `start_sec` and `end_sec` must be calculated by sequentially adding the `estimated_sec` from the script lines. The first shot always starts at 0.0.
2.  **Visualize the Script:** For each shot, carefully read the script line and caption. Imagine what this looks like visually, keeping the overall `mood` and `pacing` from the `CanonicalSpec` in mind.
3.  **Generate Rich Asset Tags:** Based on your visualization, generate a list of 3-5 rich, descriptive `desired_asset_tags` suitable for a stock footage library (e.g., "cinematic close-up of woman tired at desk sunlight", "energetic person jogging in park at sunrise").
4.  **Suggest Motion (Strictly):** For each shot, you MUST suggest a `desired_motion` by choosing ONLY from the following valid options: {motion_options}. Do not invent new motion types.
5.  **Suggest Shot Type:** For each shot, choose the `shot_type` (e.g., 'close_up', 'wide_shot') that best fits the scene.
6.  **Describe the Setting:** Provide a brief `setting_description` for the location (e.g., 'a sunlit modern kitchen', 'a busy city park at night').
7.  **Set the Time of Day:** Choose a `time_of_day` from the available options to enhance the mood and lighting.
8.  **Add Supplemental Overlays:** Your primary job is to use the `caption` from the script for on-screen text. Only use the `desired_overlays` field for supplemental graphics requested in the `CanonicalSpec`, such as a main 'title_card' (usually for the first shot) or 'stickers'. Do not create `OverlayRequest` objects for the main captions.

Your output must be a single JSON object that strictly conforms to the `ShotListOutput` schema, using the fully populated `ShotEntry` sub-model, including the `desired_overlays` field where appropriate.
"""

# --- 3. The Main Agent Logic ---
def shot_planner_logic(state: Dict[str, Any]) -> Dict[str, List[ShotEntry]]:
    """
    Converts the script into a timestamped shot list, which acts as the
    visual blueprint for the video.
    """
    spec: CanonicalSpec = state.get('spec')
    script: List[Dict[str, Any]] = [line.model_dump() for line in state.get('script',{})] # Convert Pydantic models to dicts for the prompt

    llm_user_prompt = f"""
Please create a shot list based on the following `CanonicalSpec` and `script`.

**1. The `CanonicalSpec` (Your creative constraints):**
---
{spec.model_dump_json(indent=2)}
---

**2. The `script` (Your narrative and timing guide):**
---
{json.dumps(script, indent=2)}
---
"""

    # We pass the full ShotEntry schema to the LLM via the ShotListOutput model
    llm_output = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=SHOT_PLANNER_SYSTEM_PROMPT.format(
            motion_options=list(get_args(get_args(ShotEntry.model_fields['desired_motion'].annotation)[0]))
        ),
        user_prompt=llm_user_prompt,
        output_model=ShotListOutput
    )

    shot_list_data = llm_output.get("shot_list", [])
    # Validate the output with Pydantic models
    validated_shot_list = [ShotEntry(**shot_data) for shot_data in shot_list_data]

    logger.info("--- Shot Planner Agent: Generated Shot List ---")
    for shot in validated_shot_list:
        logger.info(f"- Shot {shot.shot_id} ({shot.start_sec:.2f}s - {shot.end_sec:.2f}s): Tags: {shot.desired_asset_tags}")

    return {"shot_list": validated_shot_list}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state("reel_on_coding_and_ai.json")
    state.update(shot_planner_logic(state))
    save_app_state(state, "reel_on_coding_and_ai.json")