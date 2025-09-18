import os
import warnings
import logging
import json
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, validator

from vertexai import init
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import StreamingMode, RunConfig
from toolbox_core import ToolboxSyncClient

import constants as const

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

## Organization Context
When a user provides an organization ID, search in that organization's specific FAQ first:  
- **org_id 4**: GroundWorks - Construction/Ground services  
- **org_id 5**: Method - Business methodology  
- **org_id 6**: WL Development - Development services  
- **org_id 7**: PatientNow - Healthcare/Patient management  
- **org_id 8**: JemHR - Human resources  
- **org_id 9**: ToursByLocal - Tourism/Travel services  

## Data Sources
1. **Organization-specific FAQs** 2. **Global FAQs** ## Search Strategy
1. If an org_id is provided → search org-specific FAQs first; if no strong match, fallback to global FAQs.  
2. If no org_id is provided → search global FAQs directly.  

## Response Guidelines
- Show only the **answer content**; never mention “databases,” “configs,” or technical details  
- Keep responses **concise, precise, and context-aware** - If nothing relevant is found:  
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
                model=const.DEFAULT_VERTEX_AI_MODEL_NAME,
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
        "message": "FAQ Semantic Search API",
        "version": "1.0.0",
        "status": "running",
        "initialized": runner is not None,
        "organizations": ORG_MAPPING,
        "endpoints": {
            "POST /query": "Submit FAQ query (non-streaming)",
            "POST /stream-query": "Submit FAQ query (SSE streaming)",
            "GET /chat": "Web-based chat interface using SSE",
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


# --- NEW SSE Streaming Endpoint ---
@app.post("/stream-query")
async def stream_query(request: QueryRequest):
    """Handle streaming FAQ query with Server-Sent Events (SSE)"""

    async def stream_generator():
        try:
            # Ensure system is initialized
            if runner is None:
                await initialize_components()

            org_name = (
                ORG_MAPPING.get(request.org_id, "No Organization")
                if request.org_id
                else "No Organization"
            )
            logger.info(
                f"Streaming query: '{request.query}' for org_id: {request.org_id} ({org_name})"
            )

            # Prepare context for the model
            if request.org_id:
                query_with_context = f"Organization ID: {request.org_id} (Organization: {org_name})\nUser Query: {request.query}"
            else:
                query_with_context = (
                    f"No specific organization selected\nUser Query: {request.query}"
                )

            user_content = types.Content(
                role="user", parts=[types.Part(text=query_with_context)]
            )
            run_config = RunConfig(streaming_mode=StreamingMode.STREAMING)

            # Run the agent and stream results
            result = runner.run_async(
                session_id=session_id,
                user_id="api_user",
                new_message=user_content,
                run_config=run_config,
            )

            # Yield each part of the response as an SSE event
            has_sent_content = False
            async for event in result:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            has_sent_content = True
                            # Format as SSE: data: <json_string>\n\n
                            data = {"type": "token", "content": part.text}
                            yield f"data: {json.dumps(data)}\n\n"
                            await asyncio.sleep(
                                0.01
                            )  # Small delay to ensure chunks are sent separately

            if not has_sent_content:
                fallback_text = "I couldn't find relevant information. Please try rephrasing your question."
                fallback_data = {"type": "token", "content": fallback_text}
                yield f"data: {json.dumps(fallback_data)}\n\n"

            # Signal the end of the stream
            org_info = (
                f"Searched in: {org_name}"
                if request.org_id
                else "Searched in: Global FAQ"
            )
            end_data = {"type": "end", "org_info": org_info}
            yield f"data: {json.dumps(end_data)}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            error_data = {"type": "error", "content": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# --- Original non-streaming endpoint (kept for compatibility/testing) ---
@app.post("/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """Query FAQ with semantic search (non-streaming)"""
    try:
        if runner is None:
            await initialize_components()

        org_name = (
            ORG_MAPPING.get(request.org_id, "No Organization")
            if request.org_id
            else "No Organization"
        )
        logger.info(
            f"Received query: '{request.query}' for org_id: {request.org_id} ({org_name})"
        )

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
                        response_text += part.text

        if not response_text.strip():
            response_text = "I couldn't find relevant information for your query."

        return QueryResponse(
            answer=response_text.strip(),
            org_info=(
                f"Searched in: {org_name}"
                if request.org_id
                else "Searched in: Global FAQ"
            ),
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.get("/chat")
async def chat_ui():
    """Enhanced frontend chatbot UI with organization selection and SSE streaming"""
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
            .header { text-align: center; color: white; margin-bottom: 30px; }
            .header h1 { font-size: 2.5em; font-weight: 700; margin-bottom: 10px; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
            .header p { font-size: 1.1em; opacity: 0.9; }
            #chat-container { max-width: 700px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; display: flex; flex-direction: column; height: 90vh; max-height: 800px; }
            .controls { background: #f8f9fa; padding: 20px; border-bottom: 1px solid #e9ecef; }
            .org-selector { display: flex; align-items: center; gap: 15px; }
            .org-selector label { font-weight: 600; color: #495057; font-size: 14px; }
            #orgSelect { padding: 8px 12px; border: 2px solid #dee2e6; border-radius: 8px; font-size: 14px; background: white; color: #495057; cursor: pointer; transition: border-color 0.2s; }
            #orgSelect:focus { outline: none; border-color: #667eea; }
            #chat { flex: 1; overflow-y: auto; padding: 20px; background: white; }
            .message { margin: 15px 0; display: flex; align-items: flex-start; gap: 10px; }
            .user { justify-content: flex-end; }
            .bot { justify-content: flex-start; }
            .message-content { max-width: 80%; padding: 12px 16px; border-radius: 18px; font-size: 14px; line-height: 1.4; word-wrap: break-word; }
            .user .message-content { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-bottom-right-radius: 6px; }
            .bot .message-content { background: #f8f9fa; color: #495057; border: 1px solid #e9ecef; border-bottom-left-radius: 6px; }
            .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; flex-shrink: 0; }
            .user .avatar { background: linear-gradient(135deg, #667eea, #764ba2); color: white; order: 1; }
            .bot .avatar { background: #e9ecef; color: #6c757d; }
            #input-container { display: flex; padding: 20px; background: #f8f9fa; gap: 10px; align-items: center; border-top: 1px solid #e9ecef;}
            #query { flex: 1; padding: 12px 16px; border: 2px solid #dee2e6; border-radius: 25px; font-size: 14px; outline: none; transition: border-color 0.2s; resize: none; }
            #query:focus { border-color: #667eea; }
            #send { padding: 12px 24px; border: none; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 25px; cursor: pointer; font-size: 14px; font-weight: 600; transition: transform 0.2s, box-shadow 0.2s; }
            #send:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
            #send:active { transform: translateY(0); }
            #send:disabled { background: #adb5bd; cursor: not-allowed; }
            .org-badge { display: inline-block; background: rgba(255, 255, 255, 0.2); color: white; padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 8px; }
        </style>
    </head>
    <body>
        <div class="header"><h1>FAQ Assistant</h1><p>Get instant answers to your frequently asked questions</p></div>
        <div id="chat-container">
            <div class="controls">
                <div class="org-selector">
                    <label for="orgSelect">Organization:</label>
                    <select id="orgSelect">
                        <option value="">-- Global FAQs --</option>
                        <option value="4">GroundWorks</option><option value="5">Method</option><option value="6">WL Development</option>
                        <option value="7">PatientNow</option><option value="8">JemHR</option><option value="9">ToursByLocal</option>
                    </select>
                </div>
            </div>
            <div id="chat">
                <div class="message bot">
                    <div class="avatar">🤖</div>
                    <div class="message-content">Hello! Select an organization and ask a question. I'll search our knowledge base to find an answer for you.</div>
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
                    const orgText = orgValue ? orgSelect.options[orgSelect.selectedIndex].text : "Global";
                    content.innerHTML = text + `<span class="org-badge">${orgText}</span>`;
                } else {
                    content.textContent = text;
                }
                
                if (sender === 'bot') {
                    msgDiv.appendChild(avatar);
                    msgDiv.appendChild(content);
                } else {
                    msgDiv.appendChild(content);
                    msgDiv.appendChild(avatar);
                }

                chatBox.appendChild(msgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                return content; // Return the content element to update it later
            }

            async function sendQuery() {
                const query = queryInput.value.trim();
                if (!query) return;

                const orgId = orgSelect.value || null;

                addMessage(query, "user", true);
                queryInput.value = "";
                queryInput.focus();
                sendButton.disabled = true;

                // Add a placeholder for the bot's response
                const botMessageContent = addMessage("...", "bot");
                botMessageContent.textContent = ""; // Clear the placeholder text

                try {
                    const response = await fetch("/stream-query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: query, org_id: orgId })
                    });
                    
                    if (!response.body) {
                        throw new Error("Response body is null.");
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = "";

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\\n\\n');
                        
                        buffer = lines.pop() || ""; // Keep the last, possibly incomplete, line in buffer

                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                const jsonStr = line.substring(6);
                                const data = JSON.parse(jsonStr);
                                
                                if (data.type === 'token') {
                                    botMessageContent.textContent += data.content;
                                } else if (data.type === 'error') {
                                    botMessageContent.textContent = `Error: ${data.content}`;
                                }
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                        }
                    }

                } catch (err) {
                    console.error("Streaming error:", err);
                    botMessageContent.textContent = "Sorry, I'm having trouble connecting right now. Please try again later.";
                } finally {
                    sendButton.disabled = false;
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            }

            sendButton.addEventListener("click", sendQuery);
            queryInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendQuery();
                }
            });
            queryInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
