# faq_agent.py
from google.adk.agents.llm_agent import LlmAgent
from toolbox_core import ToolboxSyncClient
from constants import FAQ_PROMPT, TOOLBOX_URL

toolbox = ToolboxSyncClient(TOOLBOX_URL)

faq_agent = LlmAgent(
    name="FAQAgent",
    model="gemini-2.0-flash",
    instruction=FAQ_PROMPT,
    tools=toolbox.load_toolset("cloudsql_faq_analysis_tools"),
)
