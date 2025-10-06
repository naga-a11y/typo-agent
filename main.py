import logging
import warnings
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from constants import (
    PROJECT_ID, TOOLBOX_URL, ORG_MAPPING, DEFAULT_VERTEX_AI_MODEL_NAME,
    MEMORY_MYSQL_URL, PROMPT, HTML_CONTENT
)

from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import StreamingMode, RunConfig
from toolbox_core import ToolboxSyncClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faq_api")

warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str
    org_id: Optional[str] = None

    @validator("org_id", pre=True)
    def empty_string_to_none(cls, v):
        if not v:
            return None
        org_set = set(ORG_MAPPING.keys())
        if v and v not in org_set:
            raise ValueError(f"Invalid org_id: {v}.")
        return v

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"
    org_info: Optional[str] = None

# --- Helper Functions ---
def build_query_context(query: str, org_id: Optional[str]) -> str:
    if org_id:
        org_name = ORG_MAPPING.get(org_id, "No Organization")
        return f"Organization ID: {org_id} ({org_name})\nUser Query: {query}"
    return f"No specific organization selected\nUser Query: {query}"

def format_log_context(**kwargs):
    return " | ".join(f"{k}={v!r}" for k, v in kwargs.items())

# --- FastAPI app ---
app = FastAPI(title="FAQ Semantic Search API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

root_agent = None
runner = None
initialization_lock = asyncio.Lock()
session_service = DatabaseSessionService(db_url=MEMORY_MYSQL_URL)

async def get_or_create_session(user_id: str, session_id: Optional[str] = None):
    sessions = await session_service.list_sessions(app_name="faq_app", user_id=user_id)
    for s in sessions.sessions:
        if session_id and s.id == session_id:
            logger.info(f"Using existing session {s.id} for user {user_id}")
            return s.id
    new_session = await session_service.create_session(
        state={}, app_name="faq_app", user_id=user_id
    )
    logger.info(f"Created session {new_session.id} for user {user_id}")
    return new_session.id

async def initialize_components():
    global root_agent, runner
    async with initialization_lock:
        if runner is not None:
            return
        logger.info("Initializing Vertex AI for FAQ API...")
        init(project=PROJECT_ID, location="us-central1")
        toolbox = ToolboxSyncClient(TOOLBOX_URL)
        faq_tools = toolbox.load_toolset("cloudsql_faq_analysis_tools")
        root_agent = LlmAgent(
            name="FAQSemanticSearchAssistant",
            model=DEFAULT_VERTEX_AI_MODEL_NAME,
            instruction=PROMPT,
            tools=faq_tools,
        )
        runner = Runner(
            app_name="faq_app",
            agent=root_agent,
            session_service=session_service,
        )
        logger.info("Runner and components initialized successfully")

@app.on_event("startup")
async def on_startup():
    try:
        await initialize_components()
        logger.info("Application startup complete.")
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise

@app.get("/")
async def root():
    return {"message": "FAQ Semantic Search API is running."}

@app.get("/healthz")
async def health():
    return {"status": "healthy", "initialized": runner is not None}

@app.post("/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    try:
        if not runner:
            raise HTTPException(status_code=503, detail="Service not initialized")
        request_user_id = f"api_user_{request.org_id or 'global'}"
        request_session_id = await get_or_create_session(user_id=request_user_id)
        org_name = ORG_MAPPING.get(request.org_id, "No Organization")
        query_with_context = build_query_context(request.query, request.org_id)
        user_content = types.Content(role="user", parts=[types.Part(text=query_with_context)])
        run_config = RunConfig(streaming_mode=StreamingMode.NONE)
        result = runner.run_async(
            session_id=request_session_id,
            user_id=request_user_id,
            new_message=user_content,
            run_config=run_config,
        )
        response_text = ""
        async for event in result:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text
        if not response_text.strip():
            response_text = "Would you like to clarify your question? I can help with engineering management, delivery, or team effectiveness."
        return QueryResponse(
            answer=response_text.strip(),
            org_info=f"Searched in: {org_name}" if request.org_id else "Searched in: Global FAQ Database"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e} | {format_log_context(org_id=request.org_id, user_id=request_user_id)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query processing failed.")

@app.get("/chat")
async def chat_ui():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    user_id = f"ws_client_{websocket.client.host}:{websocket.client.port}"
    connection_session_id = await get_or_create_session(user_id=user_id)
    logger.info(f"WebSocket connected. Session: {connection_session_id}")
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                continue
            org_id = data.get("org_id")
            query_with_context = build_query_context(query, org_id)
            user_content = types.Content(role="user", parts=[types.Part(text=query_with_context)])
            run_config = RunConfig(streaming_mode=StreamingMode.SSE)
            result_generator = runner.run_async(
                session_id=connection_session_id,
                user_id=user_id,
                new_message=user_content,
                run_config=run_config,
            )
            await websocket.send_json({"sender": "bot", "text": "", "type": "start"})
            # --- The critical streaming fix is here! ---
            try:
                prev_text = ""
                async for event in result_generator:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                # Calculate only the NEW fragment
                                new_text = part.text[len(prev_text):]
                                if new_text:  # Only send new content
                                    await websocket.send_json({
                                        "sender": "bot",
                                        "text": new_text,
                                        "type": "chunk"
                                    })
                                prev_text = part.text
                                await asyncio.sleep(0)
            except Exception as stream_err:
                logger.error(f"run_async streaming error: {stream_err}")
                await websocket.send_json({"sender": "bot", "text": "Sorry, an error occurred while generating a response.", "type": "chunk"})
            await websocket.send_json({"sender": "bot", "type": "end"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {connection_session_id}")
    except Exception as e:
        logger.error(f"WebSocket error {e} | session={connection_session_id}", exc_info=True)
    finally:
        if websocket.application_state.value > 0:
            await websocket.close()
