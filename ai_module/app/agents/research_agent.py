# agents/research_agent.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_tavily import TavilySearch

# --- Core Imports ---
from .llm_utils import call_llm_with_json_output
from ..providers.loader import providers
from ..utils.logging import logger
from ..models.specs import QuestionAnswer, FactualResearch

class StepBackQuestions(BaseModel):
    """The output model for the question generation step."""
    questions: List[str] = Field(..., description="A list of 5-6 fundamental, step-back questions about the main topic.")

class ResearchPlan(BaseModel):
    """Defines the output structure for the ResearchPlannerAgent."""
    requires_research: bool = Field(..., description="Set to true if the topic requires external web research for facts and data. Set to false for personal stories, creative writing, or subjective topics.")
    research_topic: Optional[str] = Field(None, description="If research is required, this is the concise, well-defined topic for the research agent. Otherwise, this is null.")


tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    include_answer=True
)
def step_back_questions(topic: str) -> List[str]:
    examples = [
        {
            "topic": "The Four Fundamental Forces of Nature",
            "questions": [
                "What are the four fundamental forces of nature?",
                "What is the role and relative strength of each force?",
                "What particles mediate each fundamental force?",
                "How do the four fundamental forces govern the universe?",
                "Are scientists working on a unified theory for all forces?"
            ]
        },
        {
            "topic": "The Benefits of Intermittent Fasting",
            "questions": [
                "What is intermittent fasting and what are the most common methods?",
                "What are the primary health benefits associated with intermittent fasting?",
                "How does intermittent fasting affect metabolism and weight loss?",
                "What are the potential risks or side effects of intermittent fasting?",
                "Who should avoid intermittent fasting?"
            ]
        }
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{topic}"),
        ("ai", "{questions}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    QUESTION_GENERATION_PROMPT = f"""You are an expert researcher and educator. Your task is to take a topic for a short video and break it down into a set of 5-6 fundamental, step-back questions. These questions should uncover the core facts needed to create a comprehensive and accurate explainer on the subject. Here are a few examples:
    {few_shot_prompt}  
    """
    questions_output = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=QUESTION_GENERATION_PROMPT,
        user_prompt=topic,
        output_model=StepBackQuestions
    )
    questions = questions_output.get("questions", [])
    if not questions:
        logger.info("Could not generate research questions. Skipping research.")
        return []
    return questions

def retriever_logic(question: str):
    search_results = tavily_search_tool.invoke({"query": question})
    context = []
    for result in search_results.get('results'):
        if result.get('score')>=0.4:
            context.append(result.get('content'))
    return (search_results.get("answer"), context)

def summarize_research(topic: str, retrieved_data: List[str]) -> str:
    """
    Summarizes a list of retrieved text snippets into a coherent research summary.
    
    Args:
        topic: The main topic of the research.
        retrieved_data: A list of strings, where each string is a piece of context
                        retrieved from the search tool.
                        
    Returns:
        A single string containing the synthesized summary.
    """
    # 1. Define a clear prompt template using your techniques
    #    - Persona: "expert research analyst"
    #    - Scope: "synthesize... into a single, coherent... summary"
    #    - Boundaries: "entirely based on the provided research context"
    #    - Direction: Clear rules and input placeholders
    summarizer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert research analyst and content synthesizer.
Your task is to take a main topic and a collection of research context, then consolidate all this information into a clean, factually-accurate research summary.

**Rules:**
1.  **Synthesize a Summary:** Read through all the provided context to create a single, coherent, and concise summary of the most important facts. Do not just list points.
2.  **Be Factual:** Do not invent information. Your entire output must be grounded in the provided context.
3.  **Be Objective:** Avoid speculation or personal opinions.
4.  **Stay Focused:** Only include information that is directly relevant to the main topic: **{topic}**.
5.  **Direct Output:** Provide ONLY the summary text. Do not include any introductory phrases, commentary, or conversational text like "Here is the summary:". Just the summary itself.""",
            ),
            (
                "user",
                """Please synthesize the following research context into a summary.
                
**Research Context:**
---
{context}
---""",
            ),
    ])

    llm = providers.llm

    summarization_chain = summarizer_prompt | llm | StrOutputParser()
    
    context_string = "\n\n---\n\n".join(retrieved_data)

    summary = summarization_chain.invoke(
        {"topic": topic, "context": context_string}
    )

    return summary


def research_planner_logic(enhanced_prompt:str) -> ResearchPlan:
    """
    Analyzes the enhanced_prompt to determine if research is needed and
    what the specific research topic should be.
    """
    RESEARCH_PLANNER_SYSTEM_PROMPT = """
You are a logical Research Planner. Your job is to analyze a creative prompt for a video and decide if factual research is needed to create the script.

**Your Task:**
You will be given an 'enhanced prompt'. Your task is to determine if the prompt is about a general knowledge topic that requires web research, OR if it's a personal/subjective topic (like a travel diary, a personal story, a business announcement) that does not.

**Your Process:**
1.  **Analyze the Prompt:** Read the prompt to understand its core subject. Is it about verifiable facts (e.g., 'history of Rome', 'benefits of hydration') or personal experiences (e.g., 'my summer vacation', 'our company's new mission')?
2.  **Set `requires_research`:** Set this boolean to `true` for factual topics and `false` for personal/subjective ones.
3.  **Formulate `research_topic`:**
    * If `requires_research` is `true`, create a concise and specific research topic from the prompt. (e.g., if the prompt is about a reel on the health benefits of avocado, a good research topic is "Health benefits of avocados").
    * If `requires_research` is `false`, this field MUST be `null`.

Your output must be a single JSON object that strictly conforms to the `ResearchPlan` schema.

**Example 1 (Research Needed):**
* **Input Prompt:** "...a 15s reel about the health benefits of hydration..."
* **Your JSON Output:**
    ```json
    {
      "requires_research": true,
      "research_topic": "Health benefits of hydration"
    }
    ```

**Example 2 (No Research Needed):**
* **Input Prompt:** "...a reel showing off my travel memories from my trip to Goa last summer..."
* **Your JSON Output:**
    ```json
    {
      "requires_research": false,
      "research_topic": null
    }
    ```
"""
    research_plan = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=RESEARCH_PLANNER_SYSTEM_PROMPT,
        user_prompt=enhanced_prompt,
        output_model=ResearchPlan
    )
    research_plan = ResearchPlan(**research_plan)
    logger.info(f"--- ResearchPlannerAgent: Plan Complete ---")
    if research_plan.requires_research:
        logger.info(f"  - Research is required. Topic: '{research_plan.research_topic}'")
    else:
        logger.info("  - No web research is required for this topic.")

    return research_plan

# --- 4. The Main Agent Logic ---
def research_logic(state: Dict[str, Any]) -> Dict[str, FactualResearch]:
    """
    Performs factual research on a topic using the step-back technique.
    """
    logger.info("--- Executing: ResearchAgent ---")
    enhanced_prompt = state.get('enhanced_prompt')
    research_plan: Optional[ResearchPlan] = research_planner_logic(enhanced_prompt)
    if not research_plan or not research_plan.requires_research:
        return {"research": None}
    topic = research_plan.research_topic if research_plan.research_topic else None
    if not topic:
        logger.info("No topic provided. Skipping research.")
        return {"research": None}
    questions = step_back_questions(topic)
    if not questions:
        logger.info("Could not generate research questions. Skipping research.")
        return {"research": None}
    qa_pairs = []
    retrieved_data = []
    for q in questions:
        specific_query = f"{topic}: {q}"
        logger.info(f"  - Searching for: {specific_query}")
        answer, context = retriever_logic(specific_query)
        if not answer:
            logger.info(f"  - No answer found for: {q}")
            continue
        logger.info(f"  - Answer found: {answer}")
        retrieved_data.extend(context)
        qa_pairs.append(QuestionAnswer(question=q, answer=answer))
    if not qa_pairs:
        logger.info("Could not retrieve any data. Skipping research.")
        return {"research": None}

    summary = summarize_research(topic, retrieved_data)
    logger.info(f"  - Summary: {summary}")
    return {"research":FactualResearch(summary=summary , qa_pairs=qa_pairs)}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(research_logic(state))
    save_app_state(state)