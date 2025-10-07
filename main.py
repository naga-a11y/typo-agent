# main.py

import logging
import warnings
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from constants import (
    PROJECT_ID,
    ORG_MAPPING,
    MEMORY_MYSQL_URL,
    HTML_CONTENT,
)

from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import StreamingMode, RunConfig

from parent_agent import parent_agent  # This is your LlmAgent (with sub_agents)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

warnings.filterwarnings("ignore", message=".*non-text parts in the response.*")


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str
    org_id: Optional[str] = None

    @classmethod
    def validate_org_id(cls, v):
        if not v:
            return None
        if v not in ORG_MAPPING:
            raise ValueError(f"Invalid org_id: {v}.")
        return v


class QueryResponse(BaseModel):
    answer: str
    status: str = "success"
    org_info: Optional[str] = None


# --- FastAPI app ---
app = FastAPI(title="Parent Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
parent_runner = None
initialization_lock = asyncio.Lock()
session_service = DatabaseSessionService(db_url=MEMORY_MYSQL_URL)


# --- Session Helper ---
async def get_or_create_session(user_id: str, session_id: Optional[str] = None):
    sessions = await session_service.list_sessions(
        app_name="parent_app", user_id=user_id
    )
    for s in sessions.sessions:
        if session_id and s.id == session_id:
            logger.info(f"Using existing session {s.id} for user {user_id}")
            return s.id
    new_session = await session_service.create_session(
        state={}, app_name="parent_app", user_id=user_id
    )
    logger.info(f"Created session {new_session.id} for user {user_id}")
    return new_session.id


# --- Initialize Runner ---
async def initialize_components():
    global parent_runner
    async with initialization_lock:
        if parent_runner is not None:
            return
        logger.info("Initializing Vertex AI...")
        init(project=PROJECT_ID, location="us-central1")

        logger.info("Initializing parent agent runner...")
        parent_runner = Runner(
            app_name="parent_app",
            agent=parent_agent,
            session_service=session_service,
        )
        logger.info("Parent agent runner initialized successfully.")


# --- Startup Event ---
@app.on_event("startup")
async def on_startup():
    try:
        await initialize_components()
        logger.info("Application startup complete.")
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise


# --- Endpoints ---
@app.get("/")
async def root():
    return {"message": "Parent Agent API running."}


@app.get("/healthz")
async def health():
    return {"status": "healthy", "initialized": parent_runner is not None}


@app.get("/chat")
async def chat_ui():
    return HTMLResponse(HTML_CONTENT)


@app.post("/query", response_model=QueryResponse)
async def query_all(request: QueryRequest):
    try:
        if not parent_runner:
            raise HTTPException(status_code=503, detail="Service not initialized")

        user_id = f"api_user_{request.org_id or 'global'}"
        session_id = await get_or_create_session(user_id=user_id)

        # Build context
        if request.org_id:
            org_name = ORG_MAPPING.get(request.org_id, "No Organization")
            query_text = f"Organization ID: {request.org_id} ({org_name})\nUser Query: {request.query}"
        else:
            query_text = (
                f"No specific organization selected\nUser Query: {request.query}"
            )

        user_content = types.Content(role="user", parts=[types.Part(text=query_text)])
        run_config = RunConfig(streaming_mode=StreamingMode.NONE)

        result = parent_runner.run_async(
            session_id=session_id,
            user_id=user_id,
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
            org_info=(
                f"Searched in: {org_name}"
                if request.org_id
                else "Searched in: Global FAQ Database"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query processing failed.")


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    user_id = f"ws_client_{websocket.client.host}:{websocket.client.port}"
    session_id = await get_or_create_session(user_id=user_id)
    logger.info(f"WebSocket connected. Session: {session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            org_id = data.get("org_id")
            if not query:
                continue

            # Build context
            if org_id:
                org_name = ORG_MAPPING.get(org_id, "No Organization")
                query_text = (
                    f"Organization ID: {org_id} ({org_name})\nUser Query: {query}"
                )
            else:
                query_text = f"No specific organization selected\nUser Query: {query}"

            user_content = types.Content(
                role="user", parts=[types.Part(text=query_text)]
            )
            run_config = RunConfig(streaming_mode=StreamingMode.SSE)

            result_generator = parent_runner.run_async(
                session_id=session_id,
                user_id=user_id,
                new_message=user_content,
                run_config=run_config,
            )

            await websocket.send_json({"sender": "bot", "text": "", "type": "start"})

            try:
                prev_text = ""
                async for event in result_generator:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                new_text = part.text[len(prev_text) :]
                                if new_text:
                                    await websocket.send_json(
                                        {
                                            "sender": "bot",
                                            "text": new_text,
                                            "type": "chunk",
                                        }
                                    )
                                prev_text = part.text
                                await asyncio.sleep(0)
            except Exception as stream_err:
                logger.error(f"Streaming error: {stream_err}", exc_info=True)
                await websocket.send_json(
                    {
                        "sender": "bot",
                        "text": "Sorry, an error occurred while generating a response.",
                        "type": "chunk",
                    }
                )
            await websocket.send_json({"sender": "bot", "type": "end"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e} | session={session_id}", exc_info=True)
    finally:
        if websocket.application_state.value > 0:
            await websocket.close()
