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

# --- Initialize Vertex AI ---
init(project=const.PROJECT_ID, location="us-central1")

# --- System Prompt ---
prompt = """
# FAQ Semantic Search Assistant for org_id = {org_id}

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
- Keep answers concise and focused on the user’s query.
"""

# --- Toolbox setup ---
TOOLBOX_URL = const.TOOLBOX_URL

def get_faq_agent():
    toolbox = ToolboxSyncClient(TOOLBOX_URL)
    return LlmAgent(
        name="FAQSemanticSearchAssistant",
        model="gemini-2.5-pro",
        instruction = prompt.format(org_id=const.ORG_ID),
        tools=toolbox.load_toolset("cloudsql_faq_analysis_tools"),
    )

# --- FastAPI app ---
app = FastAPI()
session_service = InMemorySessionService()
root_agent = get_faq_agent()
runner = Runner(app_name="faq_app", agent=root_agent, session_service=session_service)

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
        </style>
    </head>
    <body>
        <h2>FAQ Semantic Search Chatbot (WebSocket)</h2>
        <div id="chat"></div>
        <br>
        <input type="text" id="query" placeholder="Ask a question...">
        <button onclick="ask()">Send</button>

        <script>
            // Use relative URL so it works on Cloud Run (wss://)
            const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
            const chatBox = document.getElementById("chat");
            let botMessageDiv = null;

            // --- Typing Queue ---
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
                        setTimeout(typing, 20); // speed: 20ms per char
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

            // --- Handle WebSocket messages ---
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

            // --- Send user message ---
            function ask() {
                const query = document.getElementById("query").value.trim();
                if (!query) return;

                const userDiv = document.createElement("div");
                userDiv.className = "user";
                userDiv.innerHTML = "<b>You:</b> " + query;
                chatBox.appendChild(userDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

                ws.send(query);

                botMessageDiv = document.createElement("div");
                botMessageDiv.className = "bot";
                botMessageDiv.innerHTML = "<b>Bot:</b> ";
                chatBox.appendChild(botMessageDiv);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Create session per connection
    session = await session_service.create_session(
        state={}, app_name="faq_app", user_id="demo_user"
    )

    while True:
        try:
            query = await websocket.receive_text()
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

        except Exception as e:
            await websocket.send_text(f"[ERROR] {str(e)}")
            break
