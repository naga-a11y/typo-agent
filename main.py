import os
import warnings
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
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

# --- System Prompt ---
prompt = """
# FAQ Semantic Search Assistant for particular org_id

You are a **FAQ Semantic Search Assistant** that helps users find relevant answers 
from FAQ data using semantic search.

## Role
- Each organization may have **custom FAQ configs** in `typo_org.faq_config`.
- Always check `faq_config` first for the given org_id.
- If no good results are found, fallback to the global FAQ database `typo_org.faq_entries`.

## Data sources
1. CloudSQL table: `typo_org.faq_config`
   - id, org_id, text, embedding, created_at
2. CloudSQL table: `typo_org.faq_entries`
   - id, text, embedding, created_at

## How to search
Tools available:
- `search_faq_config_semantic` (search org-specific config by org_id + query)
- `search_faq_entries_semantic` (fallback global FAQ search)

### Process
1. Call `search_faq_config_semantic` with { "org_id": <org_id>, "query": <user query> }.
2. If results exist and have high similarity, return top 1–3 results.
3. Otherwise, call `search_faq_entries_semantic` and return top 1–3 results.

## Response rules
- Show **only the text (definition or answer)** to the user.
- Do **NOT** show similarity score or created_at in the response.
- If nothing relevant is found, ask clarifying questions or suggest documentation.
- Keep answers concise and focused on the user's query.
"""

# --- FastAPI app ---
app = FastAPI(title="FAQ Semantic Search", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global initialization ---
root_agent = None
runner = None
session_service = InMemorySessionService()
initialization_lock = asyncio.Lock()

async def initialize_components():
    """Initialize heavy components with proper error handling"""
    global root_agent, runner
    
    async with initialization_lock:
        if runner is not None:
            return runner
            
        try:
            logger.info("Initializing Vertex AI...")
            init(project=const.PROJECT_ID, location="us-central1")
            
            logger.info("Initializing toolbox...")
            toolbox = ToolboxSyncClient(const.TOOLBOX_URL)
            
            logger.info("Creating LLM agent...")
            root_agent = LlmAgent(
                name="FAQSemanticSearchAssistant",
                model="gemini-2.5-pro",
                instruction=prompt,
                tools=toolbox.load_toolset("cloudsql_faq_analysis_tools"),
            )
            
            logger.info("Creating runner...")
            runner = Runner(app_name="faq_app", agent=root_agent, session_service=session_service)
            
            logger.info("Initialization completed successfully")
            return runner
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

# --- Startup event ---
@app.on_event("startup")
async def startup_event():
    """Initialize components at startup"""
    try:
        await initialize_components()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        # Don't raise here to allow health checks to work

# --- Health endpoint for Cloud Run ---
@app.get("/healthz")
async def health():
    """Health check endpoint"""
    try:
        # Basic health check
        return {"status": "ok", "initialized": runner is not None}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}

# --- Readiness endpoint ---
@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    try:
        if runner is None:
            await initialize_components()
        return {"status": "ready", "initialized": True}
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {"status": "not_ready", "message": str(e)}

# --- Serve HTML frontend ---
@app.get("/")
async def get():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FAQ Chatbot (WebSocket)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #chat { border: 1px solid #ccc; padding: 10px; width: 500px; height: 300px; overflow-y: auto; }
            .bot { color: blue; margin: 5px 0; }
            .user { color: green; margin: 5px 0; }
            .error { color: red; margin: 5px 0; }
            .status { color: orange; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h2>FAQ Semantic Search Chatbot (WebSocket)</h2>
        <div id="status">Connecting...</div>
        <div id="chat"></div>
        <br>
        <input type="text" id="query" placeholder="Ask a question..." disabled>
        <button onclick="ask()" id="sendBtn" disabled>Send</button>

        <script>
            let ws = null;
            const chatBox = document.getElementById("chat");
            const statusDiv = document.getElementById("status");
            const queryInput = document.getElementById("query");
            const sendBtn = document.getElementById("sendBtn");
            let botMessageDiv = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;

            let typingQueue = [];
            let isTyping = false;

            function typeFromQueue() {
                if (isTyping || typingQueue.length === 0) return;
                isTyping = true;
                const { targetDiv, text } = typingQueue.shift();
                let i = 0;
                function typing() {
                    if (i < text.length) {
                        targetDiv.innerHTML += text.charAt(i);
                        i++;
                        setTimeout(typing, 20);
                        chatBox.scrollTop = chatBox.scrollHeight;
                    } else {
                        isTyping = false;
                        typeFromQueue();
                    }
                }
                typing();
            }

            function typeText(targetDiv, text) {
                typingQueue.push({ targetDiv, text });
                typeFromQueue();
            }

            function connectWebSocket() {
                try {
                    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
                    
                    ws.onopen = function() {
                        statusDiv.innerHTML = '<span class="status">Connected!</span>';
                        queryInput.disabled = false;
                        sendBtn.disabled = false;
                        reconnectAttempts = 0;
                    };

                    ws.onmessage = function(event) {
                        if (event.data === "[END]") {
                            botMessageDiv = null;
                        } else if (event.data.startsWith("[") && event.data.endsWith("]")) {
                            const sysDiv = document.createElement("div");
                            sysDiv.className = "bot";
                            sysDiv.innerHTML = "<i>" + event.data + "</i>";
                            chatBox.appendChild(sysDiv);
                            chatBox.scrollTop = chatBox.scrollHeight;
                        } else {
                            if (!botMessageDiv) {
                                botMessageDiv = document.createElement("div");
                                botMessageDiv.className = "bot";
                                botMessageDiv.innerHTML = "<b>Bot:</b> ";
                                chatBox.appendChild(botMessageDiv);
                            }
                            typeText(botMessageDiv, event.data + " ");
                        }
                    };

                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        const errorDiv = document.createElement("div");
                        errorDiv.className = "error";
                        errorDiv.innerHTML = "Connection error occurred";
                        chatBox.appendChild(errorDiv);
                    };

                    ws.onclose = function(event) {
                        statusDiv.innerHTML = '<span class="error">Disconnected</span>';
                        queryInput.disabled = true;
                        sendBtn.disabled = true;
                        
                        if (reconnectAttempts < maxReconnectAttempts) {
                            reconnectAttempts++;
                            statusDiv.innerHTML += ` (Reconnecting... ${reconnectAttempts}/${maxReconnectAttempts})`;
                            setTimeout(connectWebSocket, 2000);
                        } else {
                            statusDiv.innerHTML += ' (Max reconnection attempts reached)';
                        }
                    };
                } catch (error) {
                    console.error('Failed to create WebSocket:', error);
                    statusDiv.innerHTML = '<span class="error">Failed to connect</span>';
                }
            }

            function ask() {
                const query = queryInput.value.trim();
                if (!query || !ws || ws.readyState !== WebSocket.OPEN) return;

                const userDiv = document.createElement("div");
                userDiv.className = "user";
                userDiv.innerHTML = "<b>You:</b> " + query;
                chatBox.appendChild(userDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

                ws.send(query);
                queryInput.value = "";

                botMessageDiv = document.createElement("div");
                botMessageDiv.className = "bot";
                botMessageDiv.innerHTML = "<b>Bot:</b> ";
                chatBox.appendChild(botMessageDiv);
            }

            queryInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    ask();
                }
            });

            // Initialize connection
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        logger.info("WebSocket connection accepted")
        
        # Ensure components are initialized
        if runner is None:
            await websocket.send_text("[Initializing system...]")
            await initialize_components()
            await websocket.send_text("[System ready]")
        
        # Create session
        session = await session_service.create_session(
            state={}, 
            app_name="faq_app", 
            user_id="demo_user"
        )
        logger.info(f"Session created: {session.id}")

        while True:
            try:
                # Set a timeout for receiving messages
                query = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
                logger.info(f"Received query: {query}")
                
                user_content = types.Content(role="user", parts=[types.Part(text=query)])
                run_config = RunConfig(streaming_mode=StreamingMode.SSE)

                events_async = runner.run_async(
                    session_id=session.id,
                    user_id=session.user_id,
                    new_message=user_content,
                    run_config=run_config,
                )

                async for event in events_async:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                await websocket.send_text(part.text)
                            elif hasattr(part, "function_call") and part.function_call:
                                await websocket.send_text(f"[Calling tool: {part.function_call.name}]")
                            elif hasattr(part, "function_response") and part.function_response:
                                await websocket.send_text("[Tool completed]")

                await websocket.send_text("[END]")
                
            except asyncio.TimeoutError:
                logger.info("WebSocket receive timeout")
                break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send_text(f"[ERROR] {str(e)}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_text(f"[SYSTEM ERROR] {str(e)}")
        except:
            pass
    finally:
        logger.info("WebSocket connection closed")

# --- Entry point for local testing ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
