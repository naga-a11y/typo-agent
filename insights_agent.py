# insights_agent.py
from google.adk.agents.llm_agent import LlmAgent
from toolbox_core import ToolboxSyncClient
from constants import INSIGHTS_PROMPT, TOOLBOX_URL, DEFAULT_VERTEX_AI_MODEL_NAME

toolbox = ToolboxSyncClient(TOOLBOX_URL)

insights_agent = LlmAgent(
    name="GithubInsightsAgent",
    model=DEFAULT_VERTEX_AI_MODEL_NAME,
    instruction=INSIGHTS_PROMPT,
    tools=toolbox.load_toolset("cloudsql_typo_agent_analysis_tools"),
)
