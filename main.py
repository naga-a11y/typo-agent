import os
import warnings
import logging
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import StreamingMode, RunConfig
from toolbox_core import ToolboxSyncClient
import constants as const
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Suppress noisy warnings ---
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str
    org_id: Optional[str] = None

    @validator("org_id", pre=True)
    def empty_string_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v


class QueryResponse(BaseModel):
    answer: str
    status: str = "success"
    org_info: str = None


# Organization mapping for reference
ORG_MAPPING = {
    "4": "GroundWorks",
    "5": "Method",
    "6": "WL Development",
    "7": "PatientNow",
    "8": "JemHR",
    "9": "ToursByLocal",
}

# --- FastAPI app ---
app = FastAPI(title="FAQ Semantic Search API", version="1.0.0")

# --- Global variables ---
root_agent = None
runner = None
initialization_lock = asyncio.Lock()
# Use a persistent session service for production
session_service = DatabaseSessionService(db_url=const.MEMORY_MYSQL_URL)


async def get_or_create_session(user_id: str, session_id: Optional[str] = None):
    """Finds an existing session or creates a new one for a given user."""
    sessions = await session_service.list_sessions(app_name="faq_app", user_id=user_id)
    for s in sessions.sessions:
        if s.id == session_id:
            logger.info(f"Using existing session {s.id} for user {user_id}")
            return s.id

    logger.info(f"Creating a new session for user {user_id}")
    new_session = await session_service.create_session(
        state={}, app_name="faq_app", user_id=user_id
    )
    return new_session.id


async def initialize_components():
    """Initializes the AI components (agent, runner) only once."""
    global root_agent, runner
    async with initialization_lock:
        if runner is not None:
            return

        logger.info("Initializing Vertex AI...")
        init(project=const.PROJECT_ID, location="us-central1")

        logger.info("Initializing toolbox...")
        toolbox = ToolboxSyncClient(const.TOOLBOX_URL)
        faq_tools = toolbox.load_toolset("cloudsql_faq_analysis_tools")

        logger.info("Creating LLM agent...")
        root_agent = LlmAgent(
            name="FAQSemanticSearchAssistant",
            model=const.DEFAULT_VERTEX_AI_MODEL_NAME,
            instruction=const.PROMPT,
            tools=faq_tools,
        )

        logger.info("Creating runner with DatabaseSessionService...")
        runner = Runner(
            app_name="faq_app",
            agent=root_agent,
            session_service=session_service,
        )
        logger.info("Initialization completed successfully")


# --- Startup event ---
@app.on_event("startup")
async def startup_event():
    """Initialize components at startup"""
    try:
        await initialize_components()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)


# --- Health and Info Endpoints ---
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {"message": "FAQ Semantic Search API is running."}


@app.get("/healthz")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "initialized": runner is not None}


# --- FAQ Query endpoint (POST) ---
@app.post("/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """Query FAQ with semantic search. Creates a new session for every query."""
    try:
        if runner is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        request_user_id = f"api_user_{request.org_id or 'global'}"
        request_session_id = await get_or_create_session(user_id=request_user_id)

        org_name = ORG_MAPPING.get(request.org_id, "No Organization")
        query_with_context = (
            f"Organization ID: {request.org_id} (Organization: {org_name})\nUser Query: {request.query}"
            if request.org_id
            else f"No specific organization selected\nUser Query: {request.query}"
        )

        user_content = types.Content(
            role="user", parts=[types.Part(text=query_with_context)]
        )
        run_config = RunConfig(streaming_mode=StreamingMode.NONE)

        # CORRECTED: The 'await' is removed from here.
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
            response_text = "I couldn't find relevant information for your query."

        return QueryResponse(
            answer=response_text.strip(),
            org_info=(
                f"Searched in: {org_name}"
                if request.org_id
                else "Searched in: Global FAQ Database"
            ),
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


# --- Chat UI and WebSocket Endpoint ---
@app.get("/chat")
async def chat_ui():
    """Serves the frontend chatbot UI"""
    return HTMLResponse(const.HTML_CONTENT)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Handles stateful, streaming chat conversations over WebSocket."""
    await websocket.accept()

    user_id = f"ws_client_{websocket.client.host}:{websocket.client.port}"
    connection_session_id = await get_or_create_session(user_id=user_id)
    logger.info(f"WebSocket connected. Using session_id: {connection_session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                continue

            org_id = data.get("org_id")
            org_name = ORG_MAPPING.get(org_id, "Global FAQ")

            # Send acknowledgment that we're processing
            await websocket.send_json({"sender": "bot", "text": "", "type": "start"})

            query_with_context = (
                f"Organization ID: {org_id} ({org_name})\nUser Query: {query}"
                if org_id
                else f"No specific organization selected\nUser Query: {query}"
            )
            user_content = types.Content(
                role="user", parts=[types.Part(text=query_with_context)]
            )
            run_config = RunConfig(streaming_mode=StreamingMode.SSE)

            result_generator = runner.run_async(
                session_id=connection_session_id,
                user_id=user_id,
                new_message=user_content,
                run_config=run_config,
            )

            # Stream each chunk immediately as it arrives
            async for event in result_generator:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            # Send each chunk immediately
                            await websocket.send_json(
                                {"sender": "bot", "text": part.text, "type": "chunk"}
                            )
                            # Small delay to ensure message is sent
                            await asyncio.sleep(0)

            # Signal completion
            await websocket.send_json({"sender": "bot", "type": "end"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {connection_session_id}")
    except Exception as e:
        logger.error(
            f"WebSocket error in session {connection_session_id}: {e}", exc_info=True
        )
    finally:
        if websocket.client_state.CONNECTED:
            await websocket.close()
