# parent_agent.py
from google.adk.agents.llm_agent import LlmAgent
from faq_agent import faq_agent
from insights_agent import insights_agent
from constants import PARENT_PROMPT, DEFAULT_VERTEX_AI_MODEL_NAME

parent_agent = LlmAgent(
    name="CoordinatorAgent",
    model=DEFAULT_VERTEX_AI_MODEL_NAME,
    instruction=PARENT_PROMPT,
    sub_agents=[faq_agent, insights_agent],
)
