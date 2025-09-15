import warnings
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import StreamingMode, RunConfig
from toolbox_core import ToolboxSyncClient
import constants as const

# --- Suppress noisy warnings ---
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

# --- Initialize Vertex AI (safe to do early) ---
init(project=const.PROJECT_ID, location="us-central1")

# --- System Prompt ---
prompt = """ ... your FAQ prompt ... """

# --- FastAPI app ---
app = FastAPI()
session_service = InMemorySessionService()

# Globals for lazy init
root_agent = None
runner = None

def init_runner():
    """Lazy initialize runner + agent."""
    global root_agent, runner
    if runner is None or root_agent is None:
        toolbox = ToolboxSyncClient(const.TOOLBOX_URL)
        root_agent = LlmAgent(
            name="FAQSemanticSearchAssistant",
            model="gemini-2.5-pro",
            instruction=prompt,
            tools=toolbox.load_toolset("cloudsql_faq_analysis_tools"),
        )
        runner = Runner(app_name="faq_app", agent=root_agent, session_service=session_service)
    return runner

# --- Serve HTML frontend ---
@app.get("/")
async def get():
    return HTMLResponse("✅ FAQ Agent is running. Use /ws WebSocket to chat.")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Ensure runner is ready
    local_runner = init_runner()

    # Create session per connection
    session = await session_service.create_session(
        state={}, app_name="faq_app", user_id="demo_user"
    )

    while True:
        try:
            query = await websocket.receive_text()
            user_content = types.Content(role="user", parts=[types.Part(text=query)])
            run_config = RunConfig(streaming_mode=StreamingMode.SSE)

            events_async = local_runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=user_content,
                run_config=run_config,
            )

            async for event in events_async:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            await websocket.send_text(part.text)
                        elif getattr(part, "function_call", None):
                            await websocket.send_text(f"[Calling tool: {part.function_call.name}]")
                        elif getattr(part, "function_response", None):
                            await websocket.send_text("[Tool completed]")

            await websocket.send_text("[END]")

        except Exception as e:
            await websocket.send_text(f"[ERROR] {str(e)}")
            break
