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

You are a **FAQ Semantic Search Assistant** that helps users find relevant answers from FAQ data using semantic search. When queries relate to engineering management, delivery, or organizational effectiveness, you also act as a seasoned Engineering Management Coach and Data-Driven Delivery Expert.

## Identity & Audience
- Act as a trusted peer to CTOs, VPs Engineering, and Directors
- Tone: analytical, direct, grounded. No fluff, no buzzwords, no vendor pitch
- Help organizations adopt evidence-based practices using DORA, DX Core Four, SPACE, and DevEx frameworks

## Operating Principles
- Evidence first. Prefer distributions (median, p25–p75, IQR) over simple averages
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
1. **Organization-specific FAQs**: `typo_org.faq_config`
   - Contains custom FAQ entries for specific organizations
   - Fields: id, org_id, text, embedding, created_at
   
2. **Global FAQ Database**: `typo_org.faq_entries`  
   - Contains general FAQ entries available to all users
   - Fields: id, text, embedding, created_at

## Search Strategy
**Tool Functions Available:**
- `search_faq_config_semantic(org_id, query)` - Search organization-specific FAQs
- `search_faq_entries_semantic(query)` - Search global FAQ database

**Search Process:**
1. **If org_id is provided (4, 5, 6, 7, 8, or 9):**
   - First call `search_faq_config_semantic` with the specific org_id and query
   - If good results found (high similarity), return top 1-3 answers
   - If no good org-specific results, fallback to `search_faq_entries_semantic`

2. **If no org_id provided:**
   - Directly call `search_faq_entries_semantic` to search global FAQ database
   - Return top 1-3 most relevant answers

## Response Structure
For engineering management queries, structure every answer as:
1) **Executive summary**: headline insight and why it matters
2) **Drivers**: likely causes and systemic patterns (not people)
3) **Risks & caveats**: data gaps, confounders, trade-offs
4) **Next data to pull**: the minimal additions to increase confidence

## Response Guidelines
- **Content Only**: Show only the FAQ text/answer content to users
- **No Metadata**: Do not display similarity scores, created_at, or embedding data
- **Concise & Clear**: Keep answers focused and relevant to the user's query
- **Evidence-Based**: Challenge surface interpretations; explain the "because → therefore" chain
- **Crisp Communication**: Use verbs over adjectives, meaningful numbers, bullets when helpful
- **System Focus**: Avoid moral language; focus on system design
- **Context Aware**: Acknowledge when searching organization-specific vs global FAQs
- **Helpful Fallback**: If no relevant answers found, suggest rephrasing or provide general guidance

## Interaction Rules
- Ask at most one clarifying question only if it prevents a wrong recommendation
- If signal is weak or data is missing, state uncertainty clearly
- Prefer verbs over adjectives. Example: "Because lead time p75 increased 28% post-release freeze, start X; expect p75 ↓ 10–15% in 2 sprints"
"""

# --- Request/Response Models ---
from typing import Optional
from pydantic import validator

class QueryRequest(BaseModel):
    query: str
    org_id: Optional[str] = None

    @validator('org_id', pre=True)
    def empty_string_to_none(cls, v):
        if v == '' or v is None:
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
    "9": "ToursByLocal"
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
            runner = Runner(app_name="faq_app", agent=root_agent, session_service=session_service)
            
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
            "GET /ready": "Readiness check"
        },
        "example_usage": {
            "POST": {
                "url": "/query",
                "body": {"query": "How to reset password?", "org_id": "5"}
            },
            "GET": {
                "url": "/query?query=How to reset password?&org_id=5"
            }
        }
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
        org_name = ORG_MAPPING.get(request.org_id, "No Organization") if request.org_id else "No Organization"
        logger.info(f"Received query: '{request.query}' for org_id: {request.org_id} ({org_name})")
        
        # Create user message with proper org_id context
        if request.org_id:
            query_with_context = f"Organization ID: {request.org_id} (Organization: {org_name})\nUser Query: {request.query}"
        else:
            query_with_context = f"No specific organization selected\nUser Query: {request.query}"
            
        user_content = types.Content(role="user", parts=[types.Part(text=query_with_context)])
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
            org_info=f"Searched in: {org_name}" if request.org_id else "Searched in: Global FAQ Database"
        )
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

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
            
            /* Markdown formatting styles */
            .bot .message-content h2,
            .bot .message-content h3,
            .bot .message-content h4,
            .bot .message-content h5 {
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
                background: #e9ecef !important;
                color: #e83e8c !important;
                padding: 2px 4px !important;
                border-radius: 3px !important;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
                font-size: 12px !important;
            }
            
            .bot .message-content ul {
                margin: 8px 0;
                padding-left: 20px;
            }
            
            .bot .message-content li {
                margin: 4px 0;
                line-height: 1.4;
            }
            
            .bot .message-content br {
                line-height: 1.6;
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

            function parseMarkdown(text) {
                // Convert markdown formatting to HTML
                let html = text
                    // Escape HTML first
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    
                    // Bold text: **text** or ***text***
                    .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong>$1</strong>')
                    .replace(/\*\*([^*]+)\*\*\*/g, '<strong>$1</strong>')
                    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                    
                    // Italic text: *text* (but not when part of **)
                    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>')
                    
                    // Code blocks: `code`
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    
                    // Headers
                    .replace(/^#### (.+)$/gm, '<h5>$1</h5>')
                    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
                    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
                    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
                    
                    // Bullet points: - item or * item
                    .replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>')
                    
                    // Numbered lists: 1. item
                    .replace(/^\s*\d+\.\s+(.+)$/gm, '<li>$1</li>')
                    
                    // Line breaks and paragraphs
                    .replace(/\n\s*\n/g, '</p><p>')
                    .replace(/\n/g, '<br>');
                
                // Wrap consecutive <li> elements in <ul>
                html = html.replace(/(<li>.*?<\/li>)(?:\s*<li>.*?<\/li>)*/g, function(match) {
                    return '<ul>' + match + '</ul>';
                });
                
                // Wrap in paragraph if not already wrapped
                if (!html.includes('<p>') && !html.includes('<h') && !html.includes('<ul>')) {
                    html = '<p>' + html + '</p>';
                }
                
                return html;
            }

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
                } else if (sender === "bot") {
                    // Parse markdown for bot responses
                    content.innerHTML = parseMarkdown(text);
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
