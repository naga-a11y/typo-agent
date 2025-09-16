import os
import warnings
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
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
# FAQ Semantic Search Assistant

You are a **FAQ Semantic Search Assistant** that helps users find relevant answers 
from FAQ data using semantic search.

## Role
- If the user asks about a **specific organization** (query includes an `org_id` or org reference), 
  then first check `typo_org.faq_config` for that organization.
- If the user does **not** mention any organization, or no org_id is provided, 
  then always answer from the global FAQ `typo_org.faq_entries`.

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
1. If the query includes an `org_id` or clearly refers to a specific organization:
   - Call `search_faq_config_semantic` with { "org_id": <org_id>, "query": <user query> }.
   - If good results exist, return the top 1–3 answers.
   - If no good results exist, fallback to `search_faq_entries_semantic`.
2. If the query does **not** mention an org or org_id:
   - Directly call `search_faq_entries_semantic` and return the top 1–3 answers.

## Response rules
- Show **only the text (definition or answer)** to the user.
- Do **NOT** show similarity scores, embeddings, or created_at fields.
- If nothing relevant is found, ask clarifying questions or suggest documentation.
- Keep answers concise, direct, and focused on the user’s query.
"""

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str
    org_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"

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
        "endpoints": {
            "POST /query": "Submit FAQ query with JSON body",
            "GET /query": "Submit FAQ query with query parameters",
            "GET /healthz": "Health check",
            "GET /ready": "Readiness check"
        },
        "example_usage": {
            "POST": {
                "url": "/query",
                "body": {"query": "How to reset password?", "org_id": "acme_corp"}
            },
            "GET": {
                "url": "/query?query=How to reset password?&org_id=acme_corp"
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
        
        logger.info(f"Received query: {request.query} for org_id: {request.org_id}")
        
        # Create user message with org_id context
        query_with_context = f"org_id: {request.org_id}\nQuery: {request.query}"
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
        
        return QueryResponse(answer=response_text.strip())
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# --- Simple GET endpoint for testing ---
@app.get("/query")
async def query_faq_get(query: str, org_id: str = "default"):
    """Simple GET endpoint for testing"""
    request = QueryRequest(query=query, org_id=org_id)
    return await query_faq(request)


@app.get("/chat")
async def chat_ui():
    """Simple frontend chatbot UI"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FAQ Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f9f9f9; margin: 0; padding: 0; }
            #chat-container { max-width: 600px; margin: 30px auto; border: 1px solid #ddd; background: #fff; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
            #chat { height: 400px; overflow-y: auto; padding: 15px; border-bottom: 1px solid #eee; }
            .message { margin: 10px 0; }
            .user { text-align: right; color: green; }
            .bot { text-align: left; color: blue; }
            #input-container { display: flex; padding: 10px; }
            #query { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
            #send { margin-left: 10px; padding: 10px 15px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; }
            #send:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="chat"></div>
            <div id="input-container">
                <input type="text" id="query" placeholder="Ask a question..." />
                <button id="send">Send</button>
            </div>
        </div>

        <script>
            const chatBox = document.getElementById("chat");
            const queryInput = document.getElementById("query");
            const sendButton = document.getElementById("send");

            function addMessage(text, sender) {
                const msgDiv = document.createElement("div");
                msgDiv.className = "message " + sender;
                msgDiv.textContent = text;
                chatBox.appendChild(msgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendQuery() {
                const query = queryInput.value.trim();
                if (!query) return;

                addMessage("You: " + query, "user");
                queryInput.value = "";

                try {
                    const response = await fetch("/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: query, org_id: "default" })
                    });
                    const data = await response.json();
                    if (data.answer) {
                        addMessage("Bot: " + data.answer, "bot");
                    } else {
                        addMessage("Bot: (No response)", "bot");
                    }
                } catch (err) {
                    addMessage("Bot: Error connecting to server", "bot");
                }
            }

            sendButton.addEventListener("click", sendQuery);
            queryInput.addEventListener("keydown", function(e) {
                if (e.key === "Enter") sendQuery();
            });
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
