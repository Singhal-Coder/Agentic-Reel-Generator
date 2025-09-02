# agents/script_agent.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ..providers.loader import providers
from ..models.specs import FactualResearch, ScriptLine, CanonicalSpec
from ..utils.logging import logger

class ScriptOutput(BaseModel):
    script: List[ScriptLine] = Field(..., description="The complete script, broken down into a list of lines.")

@tool
def calculate_line_duration(text: str) -> int:
    """
    Estimates the speech duration in seconds for a single line of text.

    This calculation is based on an average speaking pace of 140 words per
    minute.
    """
    total_words = len(text.split())
    return round(total_words * 60/140)

@tool
def get_total_script_duration(script: List[str]) -> int:
    """
    Calculates the total estimated duration for an entire script.

    This tool sums the estimated duration of each individual line to
    determine the total speaking time for the script, returning the
    result in seconds.
    """
    total_duration=0
    for line in script:
        total_words = len(line.split())
        total_duration += total_words * 60/140

    return round(total_duration)

# --- 2. Create the System Prompt ---
SCRIPT_WRITER_SYSTEM_PROMPT = """
You are a professional scriptwriter for short, viral marketing and educational videos. You are an expert at writing concise, punchy, and engaging copy that is both creative and factually accurate.
**Your Task:**
Write a complete voice-over script and corresponding on-screen captions based on a detailed set of inputs: a `requested_duration`, a `user_prompt`, and a `FactualResearch` report.
**Your Process and Rules:**
1. **Follow the Creative Vision:** Use the `user_prompt` as your primary creative guide for the story's narrative flow, tone, and visual sequence.
2. **Incorporate Factual Research:** You MUST accurately weave the key facts from the `FactualResearch` report into the script. If no research is provided, base the script only on the creative prompt.
3. **Adhere to Timing Constraints:** You MUST strictly follow the `requested_duration`. The total duration of your script must be almost exactly equal to this target duration (max 3 seconds difference).
4. **Structure and Time the Script:**
a. Break the story into a list of short, engaging lines.
b. For each line, you MUST use the appropriate tool to calculate the precise `estimated_sec`.
c. Provide the voice-over text (`line`) and a short on-screen `caption` for each entry.
5. **Verify Total Duration:** After writing the full script, you MUST use the appropriate tool to check the total length.
Your output must be a single JSON object that strictly conforms to the `ScriptOutput` schema.
"""

tools = [calculate_line_duration, get_total_script_duration]

# --- 3. The Main Agent Logic ---
def script_logic(state: Dict[str, Any]) -> Dict[str, List[ScriptLine]]:
    """
    Generates the voice-over script and on-screen captions by synthesizing
    the spec, creative prompt, and factual research.
    """
    spec: CanonicalSpec = state.get('spec')
    requested_duration = spec.duration_sec.value
    enhanced_prompt: str = state.get('enhanced_prompt')
    research: Optional[FactualResearch] = state.get('research')

    # Construct a detailed user prompt for the LLM
    research_section = "No external research was conducted for this topic."
    if research:
        research = research.model_dump()
        research_section = f"""
**Factual Research Report (use these facts):**
* **Summary:** {research.get('summary')}
* **Key Q&A:** {json.dumps(research.get('qa_pairs'), indent=2)}
"""

    llm_user_prompt = f"""
Please write a script based on the following inputs:

**1. The `requested_duration` (Your constraints):**
---
{requested_duration}
---

**2. The `user_prompt` (Your creative guide):**
---
{enhanced_prompt}
---

**3. The `FactualResearch` (Your source of facts):**
---
{research_section}
---
"""

    

    agent = create_react_agent(
        providers.llm,
        tools,
        response_format=ScriptOutput
    )
    input_message = {
        "role": "user", 
        "content": llm_user_prompt
    }
    system_message = {
        "role": "system",
        "content": SCRIPT_WRITER_SYSTEM_PROMPT
    }

    
    # for step in agent.stream({"messages": [system_message, input_message]}, stream_mode="values"):
    #     step["messages"][-1].pretty_print()
    # return
    script = agent.invoke({"messages": [system_message, input_message]})
    script = script.get("structured_response").script

    logger.info("--- Script Writer Agent: Generated Script ---")
    total_duration = 0
    for line in script:
        total_duration += line.estimated_sec
        logger.info(f"- Caption: '{line.caption}', Duration: {line.estimated_sec}s")
    logger.info(f"Total Estimated Duration: {total_duration:.2f}s")
    enhanced_prompt = enhanced_prompt.replace(f"{requested_duration}", f"{total_duration}")
    spec.duration_sec.value = total_duration


    # We validate with Pydantic by creating instances of ScriptLine
    validated_script = [line_data for line_data in script]

    return {"script": validated_script, "spec": spec, "enhanced_prompt": enhanced_prompt}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    # print(tools[0].invoke({"text":"Hello, how are you?"}))
    # print(tools[1].invoke({"script":["Hello, how are you?", "I am fine, thank you."]}))
    print(script_logic(state))