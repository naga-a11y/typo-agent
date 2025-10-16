# faq_agent.py
from google.adk.agents.llm_agent import LlmAgent
from toolbox_core import ToolboxSyncClient
from constants import FAQ_PROMPT, TOOLBOX_URL, DEFAULT_VERTEX_AI_MODEL_NAME

toolbox = ToolboxSyncClient(TOOLBOX_URL)

faq_agent = LlmAgent(
    name="FAQAgent",
    model=DEFAULT_VERTEX_AI_MODEL_NAME,
    instruction=FAQ_PROMPT,
    tools=toolbox.load_toolset("cloudsql_faq_analysis_tools"),
)
