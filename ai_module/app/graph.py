# graph.py

from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END
from pathlib import Path
from .models.specs import CanonicalSpec, ShotEntry, AssetCandidate, FactualResearch, ScriptLine
from .models.moviepy_params import MoviePyParams
from .models.render_params import FinalRenderParams

from .agents.creative_agents import prompt_enhancer_logic
from .agents.blueprint_agent import blueprint_logic
from .agents.extractor import extract_instructions
from .agents.script_agent import script_logic
from .agents.shot_planner_agent import shot_planner_logic
from .agents.research_agent import research_logic
from .agents.visual_retriever_agent import visual_retriever_logic
from .agents.subtitle_agent import subtitle_logic
from .agents.music_retriever_agent import music_retriever_logic
from .agents.sticker_retriever_agent import sticker_retriever_logic
from .agents.tts_agent import tts_logic
from .agents.timeline_builder_agent import timeline_builder_logic
from .agents.moviepy_agent import moviepy_agent_logic

class AppState(TypedDict):
    prompt: str
    params: dict
    spec: CanonicalSpec
    enhanced_prompt: str
    research: Optional[FactualResearch]
    script: List[ScriptLine]
    shot_list: List[ShotEntry]
    visual_assets: Optional[Dict[str, List[AssetCandidate]]]
    sticker_assets: Optional[Dict[str, List[AssetCandidate]]]
    music_asset: Optional[AssetCandidate]
    tts_file: Optional[AssetCandidate]
    subtitle_file: Optional[Path]
    timeline_json: MoviePyParams
    final_json: FinalRenderParams
    output_path: Optional[Path]

def instruction_extractor(state: AppState) -> AppState:
    print("--- Executing: InstructionExtractor ---")
    return extract_instructions(state)

def prompt_enhancer_agent(state: AppState) -> AppState:
    print("--- Executing: PromptEnhancerAgent ---")
    return prompt_enhancer_logic(state)

def blueprint_agent(state: AppState) -> AppState:
    print("--- Executing: BlueprintAgent ---")
    return blueprint_logic(state)

def script_writer_agent(state: AppState) -> AppState:
    print("--- Executing: ScriptWriterAgent ---")
    return script_logic(state)

def shot_planner_agent(state: AppState) -> AppState:
    print("--- Executing: ShotPlannerAgent ---")
    return shot_planner_logic(state)

def research_agent(state: AppState) -> AppState:
    print("--- Executing: ResearchAgent ---")
    return research_logic(state)

# --- Parallel Asset Retrieval Agents ---
def visual_retriever_agent(state: AppState) -> AppState:
    print("--- Executing: VisualRetrieverAgent ---")
    return visual_retriever_logic(state)

def sticker_retriever_agent(state: AppState) -> AppState:
    print("--- Executing: StickerRetrieverAgent ---")
    return sticker_retriever_logic(state)

def music_retriever_agent(state: AppState) -> AppState:
    print("--- Executing: MusicRetrieverAgent ---")
    return music_retriever_logic(state)

def tts_agent(state: AppState) -> AppState:
    print("--- Executing: TTSAgent ---")
    return tts_logic(state)

def subtitle_agent(state: AppState) -> AppState:
    print("--- Executing: SubtitleAgent ---")
    return subtitle_logic(state)

# --- Assembly Agents ---
def timeline_builder_agent(state: AppState) -> AppState:
    print("--- Executing: TimelineBuilderAgent (Joining parallel branches) ---")
    return timeline_builder_logic(state)

def moviepy_agent(state: AppState) -> AppState:
    print("--- Executing: MoviePyAgent ---")
    return moviepy_agent_logic(state)

def sticker_required_condition(state: AppState) -> str:
    rp = state.get("spec")
    if rp and rp.overlay_elements and rp.overlay_elements.value:
        for element in rp.overlay_elements.value:
            if element == "stickers":
                return "Requires Stickers"
    return "No Stickers required"

def tts_required_condition(state: AppState) -> str:
    rp = state.get("spec")
    if rp and rp.voice_style and rp.voice_style.value and rp.voice_style.value!="none":
        return "Requires TTS"
    return "No TTS required"
    

# Define the graph
workflow = StateGraph(AppState)

# Add nodes to the graph
workflow.add_node("InstructionExtractor", instruction_extractor)
workflow.add_node("PromptEnhancerAgent", prompt_enhancer_agent)
workflow.add_node("BlueprintAgent", blueprint_agent)
workflow.add_node("ResearchAgent", research_agent)
workflow.add_node("ScriptWriterAgent", script_writer_agent)
workflow.add_node("ShotPlannerAgent", shot_planner_agent)

# Parallel branch nodes
workflow.add_node("VisualRetrieverAgent", visual_retriever_agent)
workflow.add_node("StickerRetrieverAgent", sticker_retriever_agent)
workflow.add_node("MusicRetrieverAgent", music_retriever_agent)
workflow.add_node("TTSAgent", tts_agent)
workflow.add_node("SubtitleAgent", subtitle_agent)

# Joiner node
workflow.add_node("TimelineBuilderAgent", timeline_builder_agent, defer=True)
workflow.add_node("MoviePyAgent", moviepy_agent)


# === Flow Definitions ===

workflow.set_entry_point("InstructionExtractor")

# 1. Creative Core Pipeline (Sequential)
workflow.add_edge("InstructionExtractor", "PromptEnhancerAgent")
workflow.add_edge("PromptEnhancerAgent", "BlueprintAgent")
workflow.add_edge("BlueprintAgent", "ResearchAgent")
workflow.add_edge("ResearchAgent", "ScriptWriterAgent")
workflow.add_edge("ScriptWriterAgent", "ShotPlannerAgent")

# 2. Trigger all parallel asset branches from their respective nodes
workflow.add_edge("BlueprintAgent", "MusicRetrieverAgent")
workflow.add_edge("ShotPlannerAgent", "VisualRetrieverAgent")
workflow.add_edge("ShotPlannerAgent", "SubtitleAgent")

# 3. Conditional Branches
workflow.add_conditional_edges("ShotPlannerAgent", sticker_required_condition, {
    "Requires Stickers": "StickerRetrieverAgent",
    "No Stickers required": "TimelineBuilderAgent",
})
workflow.add_conditional_edges("ScriptWriterAgent", tts_required_condition, {
    "Requires TTS": "TTSAgent",
    "No TTS required": "TimelineBuilderAgent",
})

# 4. Join ALL asset branches into the TimelineBuilderAgent
workflow.add_edge("MusicRetrieverAgent", "TimelineBuilderAgent")
workflow.add_edge("VisualRetrieverAgent", "TimelineBuilderAgent")
workflow.add_edge("StickerRetrieverAgent", "TimelineBuilderAgent")
workflow.add_edge("TTSAgent", "TimelineBuilderAgent")
workflow.add_edge("SubtitleAgent", "TimelineBuilderAgent")

# 5. Final Assembly
workflow.add_edge("TimelineBuilderAgent", "MoviePyAgent")
workflow.add_edge("MoviePyAgent", END)

# Compile the graph
app = workflow.compile()