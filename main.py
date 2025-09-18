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
# FAQ Semantic Search Assistant & Engineering Management Coach

You are a **FAQ Semantic Search Assistant** that helps users find relevant answers from FAQ data using semantic search.  
When queries relate to engineering management, delivery, or organizational effectiveness, you also act as a seasoned **Engineering Management Coach and Data-Driven Delivery Expert**.

## Identity & Audience
- Act as a trusted peer to CTOs, VPs Engineering, and Directors  
- Tone: analytical, precise, direct. No fluff, no buzzwords, no vendor pitch  
- Help organizations adopt evidence-based practices using DORA, DX Core Four, SPACE, and DevEx frameworks  

## Operating Principles
- Keep answers short and precise; prioritize clarity over completeness  
- Prioritize causality over correlation; call out confounders and seasonality  
- Emphasize team-level patterns, systemic blockers, and long-term trends; avoid individual blame  
- If signal is weak or data is missing, state uncertainty clearly and specify what's needed  

## Organization Context
When a user provides an organization ID, search in that organization's specific FAQ first:  
- **org_id 4**: GroundWorks - Construction/Ground services  
- **org_id 5**: Method - Business methodology  
- **org_id 6**: WL Development - Development services  
- **org_id 7**: PatientNow - Healthcare/Patient management  
- **org_id 8**: JemHR - Human resources  
- **org_id 9**: ToursByLocal - Tourism/Travel services  

## Data Sources
1. **Organization-specific FAQs**  
2. **Global FAQs**  

## Search Strategy
1. If an org_id is provided → search org-specific FAQs first; if no strong match, fallback to global FAQs.  
2. If no org_id is provided → search global FAQs directly.  

## Response Guidelines
- Show only the **answer content**; never mention "databases," "configs," or technical details  
- Keep responses **concise, precise, and context-aware**  
- If nothing relevant is found:  
  - Do **not** say "I didn't find anything"  
  - Instead, provide a short, helpful fallback (e.g., ask a clarifying question, or give a general engineering/delivery insight if relevant)  
- For **out-of-scope queries** (like "what is today's date?"): reply politely with  
  `"Sorry, I don't have info about that, but I can help you with engineering management, delivery, or organizational effectiveness."`  
- If signal is weak: acknowledge uncertainty clearly and point out what more is needed  
- Prefer verbs over adjectives. Example:  
  `"Because lead time p75 increased 28% post-release freeze, start X; expect p75 ↓ 10–15% in 2 sprints."`  

## Interaction Rules
- Ask at most one clarifying question only if it prevents a wrong recommendation  
- Never reveal implementation details (FAQ configs, embeddings, similarity scores, etc.)  
"""

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
session_service = InMemorySessionService()
session_id = None
initialization_lock = asyncio.Lock()

async def initialize_components():
    """Initialize components with error handling"""
    global root_agent, runner, session_id

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
            runner = Runner(
                app_name="faq_app", agent=root_agent, session_service=session_service
            )

            # Create a persistent session
            session = await session_service.create_session(
                state={}, app_name="faq_app", user_id="api_user"
            )
            session_id = session.id

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

# --- Health endpoints ---
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "FAQ Semantic Search API with WebSocket Streaming",
        "version": "1.0.0",
        "status": "running",
        "initialized": runner is not None,
        "organizations": ORG_MAPPING,
        "endpoints": {
            "POST /query": "Submit FAQ query with JSON body",
            "GET /query": "Submit FAQ query with query parameters", 
            "WS /ws": "WebSocket streaming endpoint",
            "GET /chat": "Web-based chat interface",
            "GET /healthz": "Health check",
            "GET /ready": "Readiness check",
        },
    }

@app.get("/healthz")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "initialized": runner is not None}

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

# --- WebSocket endpoint for streaming ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming responses"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Ensure system is initialized
        if runner is None:
            await websocket.send_text(json.dumps({"type": "system", "content": "Initializing system..."}))
            await initialize_components()
            await websocket.send_text(json.dumps({"type": "system", "content": "System ready!"}))

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                query = message_data.get("query", "").strip()
                org_id = message_data.get("org_id")
                
                if not query:
                    continue
                
                # Convert empty string to None
                if org_id == "":
                    org_id = None
                
                # Get organization info
                org_name = ORG_MAPPING.get(org_id, "No Organization") if org_id else "No Organization"
                logger.info(f"WebSocket query: '{query}' for org_id: {org_id} ({org_name})")
                
                # Send acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "ack", 
                    "content": f"Searching{' in ' + org_name if org_id else ' in Global FAQ'}..."
                }))
                
                # Create user message with context
                if org_id:
                    query_with_context = f"Organization ID: {org_id} (Organization: {org_name})\nUser Query: {query}"
                else:
                    query_with_context = f"No specific organization selected\nUser Query: {query}"
                
                user_content = types.Content(
                    role="user", parts=[types.Part(text=query_with_context)]
                )
                run_config = RunConfig(streaming_mode=StreamingMode.SSE)
                
                # Run the query with streaming
                result = runner.run_async(
                    session_id=session_id,
                    user_id="api_user",
                    new_message=user_content,
                    run_config=run_config,
                )
                
                response_chunks = []
                async for event in result:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                # Send streaming chunk
                                await websocket.send_text(json.dumps({
                                    "type": "chunk",
                                    "content": part.text
                                }))
                                response_chunks.append(part.text)
                            elif hasattr(part, "function_call") and part.function_call:
                                # Send function call notification
                                await websocket.send_text(json.dumps({
                                    "type": "function",
                                    "content": f"🔍 Searching {part.function_call.name}..."
                                }))
                            elif hasattr(part, "function_response") and part.function_response:
                                # Send function completion notification
                                await websocket.send_text(json.dumps({
                                    "type": "function",
                                    "content": "✓ Search completed"
                                }))
                
                # Send completion signal
                full_response = " ".join(response_chunks)
                if not full_response.strip():
                    full_response = "I couldn't find relevant information for your query. Please try rephrasing your question."
                
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "content": full_response,
                    "org_info": f"Searched in: {org_name}" if org_id else "Searched in: Global FAQ Database"
                }))
                
                logger.info(f"WebSocket response completed: {full_response[:100]}...")
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Invalid message format"
                }))
            except Exception as e:
                logger.error(f"WebSocket query error: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Query processing failed: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection terminated")

# --- REST API endpoints (keep for backward compatibility) ---
@app.post("/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """Query FAQ with semantic search (REST API)"""
    try:
        if runner is None:
            await initialize_components()

        org_name = ORG_MAPPING.get(request.org_id, "No Organization") if request.org_id else "No Organization"
        logger.info(f"REST query: '{request.query}' for org_id: {request.org_id} ({org_name})")

        if request.org_id:
            query_with_context = f"Organization ID: {request.org_id} (Organization: {org_name})\nUser Query: {request.query}"
        else:
            query_with_context = f"No specific organization selected\nUser Query: {request.query}"

        user_content = types.Content(role="user", parts=[types.Part(text=query_with_context)])
        run_config = RunConfig(streaming_mode=StreamingMode.NONE)

        result = runner.run_async(
            session_id=session_id,
            user_id="api_user",
            new_message=user_content,
            run_config=run_config,
        )

        response_text = ""
        async for event in result:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text + " "

        if not response_text.strip():
            response_text = "I couldn't find relevant information for your query. Please try rephrasing your question."

        return QueryResponse(
            answer=response_text.strip(),
            org_info=f"Searched in: {org_name}" if request.org_id else "Searched in: Global FAQ Database",
        )

    except Exception as e:
        logger.error(f"REST query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/query")
async def query_faq_get(query: str, org_id: str = None):
    """Simple GET endpoint for testing"""
    request = QueryRequest(query=query, org_id=org_id)
    return await query_faq(request)

@app.get("/chat")
async def chat_ui():
    """Enhanced frontend chatbot UI with WebSocket streaming"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FAQ Chatbot - Streaming</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            
            .header h1 {
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            #chat-container { 
                max-width: 700px; 
                margin: 0 auto; 
                background: white;
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            
            .controls {
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #e9ecef;
            }
            
            .org-selector {
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            .org-selector label {
                font-weight: 600;
                color: #495057;
                font-size: 14px;
            }
            
            #orgSelect {
                padding: 8px 12px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                font-size: 14px;
                background: white;
                color: #495057;
                cursor: pointer;
                transition: border-color 0.2s;
                min-width: 200px;
            }
            
            #orgSelect:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .connection-status {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                margin-left: auto;
            }
            
            .connected { background: #d4edda; color: #155724; }
            .connecting { background: #fff3cd; color: #856404; }
            .disconnected { background: #f8d7da; color: #721c24; }
            
            #chat { 
                height: 450px; 
                overflow-y: auto; 
                padding: 20px;
                background: white;
            }
            
            .message { 
                margin: 15px 0;
                display: flex;
                align-items: flex-start;
                gap: 10px;
                animation: fadeIn 0.3s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .user { justify-content: flex-end; }
            .bot { justify-content: flex-start; }
            
            .message-content {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 18px;
                font-size: 14px;
                line-height: 1.4;
                word-wrap: break-word;
            }
            
            .user .message-content {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border-bottom-right-radius: 6px;
            }
            
            .bot .message-content {
                background: #f8f9fa;
                color: #495057;
                border: 1px solid #e9ecef;
                border-bottom-left-radius: 6px;
            }
            
            .system .message-content {
                background: #e3f2fd;
                color: #1565c0;
                font-style: italic;
                font-size: 13px;
            }
            
            .function .message-content {
                background: #f3e5f5;
                color: #7b1fa2;
                font-size: 12px;
                font-style: italic;
            }
            
            .avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
                flex-shrink: 0;
            }
            
            .user .avatar {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                order: 1;
            }
            
            .bot .avatar, .system .avatar, .function .avatar {
                background: #e9ecef;
                color: #6c757d;
            }
            
            .streaming {
                border-right: 2px solid #667eea;
                animation: blink 1s infinite;
            }
            
            @keyframes blink {
                50% { border-color: transparent; }
            }
            
            #input-container { 
                display: flex; 
                padding: 20px;
                background: #f8f9fa;
                gap: 10px;
                align-items: center;
            }
            
            #query { 
                flex: 1; 
                padding: 12px 16px;
                border: 2px solid #dee2e6;
                border-radius: 25px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s;
            }
            
            #query:focus {
                border-color: #667eea;
            }
            
            #send { 
                padding: 12px 24px;
                border: none;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border-radius: 25px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            #send:hover:not(:disabled) {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            #send:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .org-badge {
                display: inline-block;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                margin-left: 8px;
            }
            
            /* Markdown formatting styles */
            .bot .message-content h2, .bot .message-content h3, .bot .message-content h4, .bot .message-content h5 {
                color: #495057;
                margin: 12px 0 6px 0;
                font-weight: 600;
            }
            
            .bot .message-content h2 { font-size: 16px; }
            .bot .message-content h3 { font-size: 15px; }
            .bot .message-content h4 { font-size: 14px; }
            .bot .message-content h5 { font-size: 13px; }
            
            .bot .message-content p {
                margin: 6px 0;
                line-height: 1.5;
            }
            
            .bot .message-content strong {
                color: #343a40;
                font-weight: 600;
            }
            
            .bot .message-content em {
                font-style: italic;
                color: #6c757d;
            }
            
            .bot .message-content code {
                background: #e9ecef;
                color: #e83e8c;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 12px;
            }
            
            .bot .message-content ul {
                margin: 8px 0;
                padding-left: 20px;
            }
            
            .bot .message-content li {
                margin: 4px 0;
                line-height: 1.4;
            }
            
            /* Custom scrollbar */
            #chat::-webkit-scrollbar {
                width: 6px;
            }
            
            #chat::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            
            #chat::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 3px;
            }
            
            #chat::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>FAQ Assistant</h1>
            <p>Get instant answers with real-time streaming</p>
        </div>

        <div id="chat-container">
            <div class="controls">
                <div class="org-selector">
                    <label for="orgSelect">Organization:</label>
                    <select id="orgSelect">
                        <option value="">-- Select Organization --</option>
                        <option value="4">GroundWorks</option>
                        <option value="5">Method</option>
                        <option value="6">WL Development</option>
                        <option value="7">PatientNow</option>
                        <option value="8">JemHR</option>
                        <option value="9">ToursByLocal</option>
                    </select>
                    <div id="connectionStatus" class="connection-status connecting">Connecting...</div>
                </div>
            </div>
            
            <div id="chat">
                <div class="message bot">
                    <div class="avatar">🤖</div>
                    <div class="message-content">
                        Hello! I'm your FAQ Assistant with real-time streaming. Select an organization above and ask me any question. You'll see my responses appear as I generate them!
                    </div>
                </div>
            </div>
            
            <div id="input-container">
                <input type="text" id="query" placeholder="Type your question here..." disabled />
                <button id="send" disabled>Send</button>
            </div>
        </div>

        <script>
            const chatBox = document.getElementById("chat");
            const queryInput = document.getElementById("query");
            const sendButton = document.getElementById("send");
            const orgSelect = document.getElementById("orgSelect");
            const connectionStatus = document.getElementById("connectionStatus");

            let ws = null;
            let currentStreamingMessage = null;
            let isConnected = false;

            function parseMarkdown(text) {
                let html = text
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/\\*\\*\\*([^*\\n]+)\\*\\*\\*/g, '<strong>$1</strong>')
                    .replace(/\\*\\*([^*\\n]+)\\*\\*/g, '<strong>$1</strong>')
                    .replace(/\\*([^*\\n]+)\\*/g, '<em>$1</em>')
                    .replace(/`([^`\\n]+)`/g, '<code>$1</code>')
                    .replace(/^#### (.+)$/gm, '<h5>$1</h5>')
                    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
                    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
                    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
                    .replace(/^\\s*[-•]\\s+(.+)$/gm, '<li>$1</li>')
                    .replace(/^\\s*\\d+\\.\\s+(.+)$/gm, '<li>$1</li>')
                    .replace(/\\n\\s*\\n/g, '</p><p>')
                    .replace(/\\n/g, '<br>');
                
                html = html.replace(/(<li>.*?<\\/li>(?:\\s*<br>\\s*<li>.*?<\\/li>)*)/g, function(match) {
                    return '<ul>' + match.replace(/<br>\\s*(?=<li>)/g, '') + '</ul>';
                });
                
                if (!html.includes('<p>') && !html.includes('<h') && !html.includes('<ul>')) {
                    html = '<p>' + html + '</p>';
                }
                
                return html;
            }

            function updateConnectionStatus(status) {
                connectionStatus.className = `connection-status ${status}`;
                switch(status) {
                    case 'connected':
                        connectionStatus.textContent = 'Connected';
                        queryInput.disabled = false;
                        sendButton.disabled = false;
                        break;
                    case 'error':
                        if (currentStreamingMessage) {
                            currentStreamingMessage.classList.remove('streaming');
                            currentStreamingMessage.innerHTML = `<span style="color: #dc3545;">${data.content}</span>`;
                            currentStreamingMessage = null;
                        } else {
                            addMessage(data.content, 'bot', false, true);
                        }
                        sendButton.disabled = false;
                        break;
                }
            }

            function addMessage(text, sender, showOrgBadge = false, isError = false) {
                const msgDiv = document.createElement("div");
                msgDiv.className = "message " + sender;
                
                const avatar = document.createElement("div");
                avatar.className = "avatar";
                
                switch(sender) {
                    case 'user':
                        avatar.textContent = "👤";
                        break;
                    case 'system':
                        avatar.textContent = "⚙️";
                        break;
                    case 'function':
                        avatar.textContent = "🔍";
                        break;
                    default:
                        avatar.textContent = "🤖";
                }
                
                const content = document.createElement("div");
                content.className = "message-content";
                
                if (sender === "user" && showOrgBadge) {
                    const orgValue = orgSelect.value;
                    const orgText = orgValue ? orgSelect.options[orgSelect.selectedIndex].text : "Global FAQ";
                    content.innerHTML = `${text} <span class="org-badge">${orgText}</span>`;
                } else if (sender === "bot" && !isError) {
                    content.innerHTML = parseMarkdown(text);
                } else {
                    content.textContent = text;
                    if (isError) {
                        content.style.color = '#dc3545';
                    }
                }
                
                msgDiv.appendChild(avatar);
                msgDiv.appendChild(content);
                chatBox.appendChild(msgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                
                return content;
            }

            function startStreamingMessage() {
                const msgDiv = document.createElement("div");
                msgDiv.className = "message bot";
                
                const avatar = document.createElement("div");
                avatar.className = "avatar";
                avatar.textContent = "🤖";
                
                const content = document.createElement("div");
                content.className = "message-content streaming";
                
                msgDiv.appendChild(avatar);
                msgDiv.appendChild(content);
                chatBox.appendChild(msgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                
                return content;
            }

            function sendQuery() {
                const query = queryInput.value.trim();
                if (!query || !isConnected) return;

                const selectedOrg = orgSelect.value;
                const orgId = selectedOrg || null;

                // Add user message
                addMessage(query, "user", true);
                queryInput.value = "";
                sendButton.disabled = true;

                // Start streaming message
                currentStreamingMessage = startStreamingMessage();

                // Send via WebSocket
                const message = {
                    query: query,
                    org_id: orgId
                };

                try {
                    ws.send(JSON.stringify(message));
                } catch (error) {
                    console.error('Failed to send message:', error);
                    addMessage("Failed to send message. Please check your connection.", "bot", false, true);
                    sendButton.disabled = false;
                    currentStreamingMessage = null;
                }
            }

            // Event listeners
            sendButton.addEventListener("click", sendQuery);
            
            queryInput.addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendQuery();
                }
            });

            // Initialize WebSocket connection
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

# --- Entry point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)d = false;
                        isConnected = true;
                        break;
                    case 'connecting':
                        connectionStatus.textContent = 'Connecting...';
                        queryInput.disabled = true;
                        sendButton.disabled = true;
                        isConnected = false;
                        break;
                    case 'disconnected':
                        connectionStatus.textContent = 'Disconnected';
                        queryInput.disabled = true;
                        sendButton.disabled = true;
                        isConnected = false;
                        break;
                }
            }

            function connectWebSocket() {
                updateConnectionStatus('connecting');
                
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

                ws.onopen = function() {
                    console.log('WebSocket connected');
                    updateConnectionStatus('connected');
                    queryInput.focus();
                };

                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (e) {
                        console.error('Failed to parse WebSocket message:', e);
                    }
                };

                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    updateConnectionStatus('disconnected');
                    setTimeout(connectWebSocket, 3000);
                };

                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    updateConnectionStatus('disconnected');
                };
            }

            function handleWebSocketMessage(data) {
                switch(data.type) {
                    case 'system':
                        addMessage(data.content, 'system');
                        break;
                    case 'ack':
                        addMessage(data.content, 'system');
                        break;
                    case 'function':
                        addMessage(data.content, 'function');
                        break;
                    case 'chunk':
                        if (currentStreamingMessage) {
                            currentStreamingMessage.innerHTML += data.content;
                            chatBox.scrollTop = chatBox.scrollHeight;
                        }
                        break;
                    case 'complete':
                        if (currentStreamingMessage) {
                            currentStreamingMessage.classList.remove('streaming');
                            currentStreamingMessage.innerHTML = parseMarkdown(data.content);
                            currentStreamingMessage = null;
                        }
                        sendButton.disable
