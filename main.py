import os
import warnings
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
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
- If signal is weak or data is missing, state uncertainty clearly and specify what’s needed  
- Maintain context and ensure responses are actionable and well-formatted.

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
- Show only the **answer content**; never mention “databases,” “configs,” or technical details  
- Keep responses **concise, precise, and context-aware**  
- If nothing relevant is found:  
  - Do **not** say “I didn’t find anything”  
  - Instead, provide a short, helpful fallback (e.g., ask a clarifying question, or give a general engineering/delivery insight if relevant)  
- For **out-of-scope queries** (like “what is today’s date?”): reply politely with  
  `"Sorry, I don’t have info about that, but I can help you with engineering management, delivery, or organizational effectiveness."`  
- If signal is weak: acknowledge uncertainty clearly and point out what more is needed  
- Prefer verbs over adjectives. Example:  
  `"Because lead time p75 increased 28% post-release freeze, start X; expect p75 ↓ 10–15% in 2 sprints."`  

## Interaction Rules
- Ask at most one clarifying question only if it prevents a wrong recommendation  
- Never reveal implementation details (FAQ configs, embeddings, similarity scores, etc.)  
"""

# --- Request/Response Models ---
from typing import Optional
from pydantic import validator


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
    org_info: str = None  # Add org info for debugging


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
session_id = None
initialization_lock = asyncio.Lock()
session_service = DatabaseSessionService(db_url=const.MEMORY_MYSQL_URL)


async def get_or_create_session(user_id: str, session_id: Optional[str] = None):
    sessions = await session_service.list_sessions(app_name="faq_app", user_id=user_id)
    for s in sessions.sessions:
        if s.id == session_id:
            logger.info(f"Using existing session {s.id}")
            return s.id

    # else create new
    new_session = await session_service.create_session(
        state={}, app_name="faq_app", user_id=user_id, session_id=session_id
    )
    return new_session.id


async def initialize_components():
    global root_agent, runner, session_id

    async with initialization_lock:
        if runner is not None:
            return runner

        logger.info("Initializing Vertex AI...")
        init(project=const.PROJECT_ID, location="us-central1")

        logger.info("Initializing toolbox...")
        toolbox = ToolboxSyncClient(const.TOOLBOX_URL)
        faq_tools = toolbox.load_toolset("cloudsql_faq_analysis_tools")

        logger.info("Creating LLM agent...")
        root_agent = LlmAgent(
            name="FAQSemanticSearchAssistant",
            model=const.DEFAULT_VERTEX_AI_MODEL_NAME,
            instruction=prompt,
            tools=faq_tools,
        )

        logger.info("Creating runner with DatabaseSessionService...")
        runner = Runner(
            app_name="faq_app",
            agent=root_agent,
            session_service=session_service,  # persistent sessions
        )

        # Create/reuse persistent session
        session_id = await get_or_create_session(user_id="api_user")

        logger.info("Initialization completed successfully")
        return runner


# --- Startup event ---
@app.on_event("startup")
async def startup_event():
    """Initialize components at startup"""
    try:
        await initialize_components()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        # Continue to allow health checks


# --- Health endpoints ---
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "FAQ Semantic Search API",
        "version": "1.0.0",
        "status": "running",
        "initialized": runner is not None,
        "organizations": ORG_MAPPING,
        "endpoints": {
            "POST /query": "Submit FAQ query with JSON body",
            "GET /query": "Submit FAQ query with query parameters",
            "GET /chat": "Web-based chat interface",
            "GET /healthz": "Health check",
            "GET /ready": "Readiness check",
        },
        "example_usage": {
            "POST": {
                "url": "/query",
                "body": {"query": "How to reset password?", "org_id": "5"},
            },
            "GET": {"url": "/query?query=How to reset password?&org_id=5"},
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


# --- FAQ Query endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """Query FAQ with semantic search"""
    try:
        # Ensure system is initialized
        if runner is None:
            await initialize_components()

        # Get organization info
        org_name = (
            ORG_MAPPING.get(request.org_id, "No Organization")
            if request.org_id
            else "No Organization"
        )
        logger.info(
            f"Received query: '{request.query}' for org_id: {request.org_id} ({org_name})"
        )

        # Create user message with proper org_id context
        if request.org_id:
            query_with_context = f"Organization ID: {request.org_id} (Organization: {org_name})\nUser Query: {request.query}"
        else:
            query_with_context = (
                f"No specific organization selected\nUser Query: {request.query}"
            )

        user_content = types.Content(
            role="user", parts=[types.Part(text=query_with_context)]
        )
        run_config = RunConfig(streaming_mode=StreamingMode.NONE)

        # Run the query
        result = runner.run_async(
            session_id=session_id,
            user_id="api_user",
            new_message=user_content,
            run_config=run_config,
        )

        # Extract the response
        response_text = ""
        async for event in result:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text + " "

        if not response_text.strip():
            response_text = "I couldn't find relevant information for your query. Please try rephrasing your question or check the documentation."

        logger.info(f"Generated response: {response_text[:100]}...")

        return QueryResponse(
            answer=response_text.strip(),
            org_info=(
                f"Searched in: {org_name}"
                if request.org_id
                else "Searched in: Global FAQ Database"
            ),
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


# --- Simple GET endpoint for testing ---
@app.get("/query")
async def query_faq_get(query: str, org_id: str = None):
    """Simple GET endpoint for testing"""
    request = QueryRequest(query=query, org_id=org_id)
    return await query_faq(request)


@app.get("/chat")
async def chat_ui():
    """Enhanced frontend chatbot UI with organization selection"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FAQ Chatbot</title>
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
            }
            
            #orgSelect:focus {
                outline: none;
                border-color: #667eea;
            }
            
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
            }
            
            .user { 
                justify-content: flex-end;
            }
            
            .bot {
                justify-content: flex-start;
            }
            
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
            
            .bot .avatar {
                background: #e9ecef;
                color: #6c757d;
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
            
            #send:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            #send:active {
                transform: translateY(0);
            }
            
            .typing {
                display: none;
                padding: 10px 0;
                font-style: italic;
                color: #6c757d;
                font-size: 13px;
            }
            
            .typing.show {
                display: block;
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
            <p>Get instant answers to your frequently asked questions</p>
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
                </div>
            </div>
            
            <div id="chat">
                <div class="message bot">
                    <div class="avatar">🤖</div>
                    <div class="message-content">
                        Hello! I'm your FAQ Assistant. Select an organization above and ask me any question. I'll search through the knowledge base to help you find answers.
                    </div>
                </div>
            </div>
            
            <div class="typing">
                <div class="message bot">
                    <div class="avatar">🤖</div>
                    <div class="message-content">Bot is typing...</div>
                </div>
            </div>
            
            <div id="input-container">
                <input type="text" id="query" placeholder="Type your question here..." />
                <button id="send">Send</button>
            </div>
        </div>

        <script>
            const chatBox = document.getElementById("chat");
            const queryInput = document.getElementById("query");
            const sendButton = document.getElementById("send");
            const orgSelect = document.getElementById("orgSelect");
            const typingIndicator = document.querySelector(".typing");

            function addMessage(text, sender, showOrgBadge = false) {
                const msgDiv = document.createElement("div");
                msgDiv.className = "message " + sender;
                
                const avatar = document.createElement("div");
                avatar.className = "avatar";
                avatar.textContent = sender === "user" ? "👤" : "🤖";
                
                const content = document.createElement("div");
                content.className = "message-content";
                
                if (sender === "user" && showOrgBadge) {
                    const orgValue = orgSelect.value;
                    const orgText = orgValue ? orgSelect.options[orgSelect.selectedIndex].text : "Global FAQ";
                    content.innerHTML = text + `<span class="org-badge">${orgText}</span>`;
                } else {
                    content.textContent = text;
                }
                
                msgDiv.appendChild(avatar);
                msgDiv.appendChild(content);
                chatBox.appendChild(msgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function showTyping() {
                typingIndicator.classList.add("show");
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function hideTyping() {
                typingIndicator.classList.remove("show");
            }

            async function sendQuery() {
                const query = queryInput.value.trim();
                if (!query) return;

                const selectedOrg = orgSelect.value;
                const orgId = selectedOrg || null;  // Convert empty string to null

                // Add user message with org badge
                addMessage(query, "user", true);
                queryInput.value = "";
                
                // Show typing indicator
                showTyping();

                try {
                    const requestBody = {
                        query: query,
                        org_id: orgId  // This will be null if no org selected
                    };

                    const response = await fetch("/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(requestBody)
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer, "bot");
                    } else if (data.detail) {
                        addMessage("Error: " + data.detail, "bot");
                    } else {
                        addMessage("I couldn't find an answer to your question. Please try rephrasing or contact support.", "bot");
                    }
                } catch (err) {
                    hideTyping();
                    addMessage("Sorry, I'm having trouble connecting right now. Please try again later.", "bot");
                }
            }

            sendButton.addEventListener("click", sendQuery);
            
            queryInput.addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendQuery();
                }
            });

            // Focus on input when page loads
            queryInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
