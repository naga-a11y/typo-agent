import os
from urllib.parse import quote_plus


def getenv(env_var, fallback=""):
    return os.getenv(env_var, fallback).strip()


# --- Environment configuration ---
ENV_MODES = {"dev", "staging", "production"}
DEFAULT_ENV = "dev"
env = getenv("ENV", DEFAULT_ENV).lower()
ENV = env if env in ENV_MODES else DEFAULT_ENV

GOOGLE_ACCESS_TOKEN = getenv("GOOGLE_ACCESS_TOKEN")

PROJECT_ID = getenv("PROJECT_ID", "typoapp-442017")
TOOLBOX_URL = getenv("TOOLBOX_URL", "https://toolbox-aua232uyqa-uc.a.run.app")
ORG_ID = int(getenv("ORG_ID", "5"))
DEFAULT_VERTEX_AI_MODEL_NAME = getenv(
    "DEFAULT_VERTEX_AI_MODEL_NAME", "gemini-2.5-pro"
)

# --- MySQL DATABASE CONFIGURATION ---
MEMORY_DB_CONFIG = {
    "DB_USER": getenv("DB_USER"),
    "DB_PASSWORD": getenv("DB_PASSWORD"),
    "HOST": getenv("HOST"),
    "MEMORY_DB_NAME": getenv("MEMORY_DB_NAME"),
}

if not all(MEMORY_DB_CONFIG.values()):
    raise RuntimeError("Environment variables for DB config are missing!")

encoded_password = quote_plus(MEMORY_DB_CONFIG["DB_PASSWORD"])

MEMORY_MYSQL_URL = (
    f"mysql+pymysql://{MEMORY_DB_CONFIG['DB_USER']}:"
    f"{encoded_password}@"
    f"{MEMORY_DB_CONFIG['HOST']}/{MEMORY_DB_CONFIG['MEMORY_DB_NAME']}"
)

# --- Org Mapping ---
ORG_MAPPING = {
    "64a82f7170940b43db25c557": "GroundWorks",
    "641b549b0bf3107d660dc743": "Method",
    "6": "WL Development",
    "653fdd265cac3a6bd3787ac0": "PatientNow",
    "8": "JemHR",
    "9": "ToursByLocal",
}

# --- System Prompt & HTML ---
PARENT_PROMPT = """
# 🧠 Coordinator Agent

You are the **Coordinator Agent** that decides which specialized sub-agent should handle a user’s request.

## Sub-agent Routing Logic

### 📚 FAQAgent
Use when the user asks about:
- Organization or product **policies**, **terms**, **FAQs**, or **configurations**
- General **how-to**, **definitions**, or **procedural questions**
- Examples:
  - "What is merge time?"
  - "How do I configure Slack?"
  - "What does burnout mean?"

### 📊 GithubInsightsAgent
Use when the user asks about:
- **Data**, **metrics**, **trends**, or **analytics**
- **GitHub activity**, **team performance**, **engineering KPIs**
- **Lists**, **comparisons**, or **insights** from organizational data
- Examples:
  - "Show PR cycle time for team X"
  - "List users in multiple orgs"
  - "Which org has the highest deployment frequency?"

---

## 💬 Greeting and Small Talk Handling
If the user says things like:
- "Hi", "Hello", "Hey", "Good morning", etc.
- or any polite greeting or thank-you message  
→ Respond briefly and naturally (e.g., “Hi there! How can I help you today?”).  
Do **not** route these messages to any sub-agent.

---

## 🚦 Routing Rules
- If the question is about **what something means** or **how something works** → use **FAQAgent**
- If the question is about **who**, **what data**, **how many**, or **what trends** → use **GithubInsightsAgent**
- Return only the **sub-agent’s response**, without adding extra commentary or explanation.
- Never mention tools, databases, internal logic, queries, or system processes.
- Never reveal internal identifiers such as organization IDs, User IDs, table names, field names, or query details.
- Always summarize responses in **user-friendly, non-technical language** focused on insights or results.
"""


INSIGHTS_PROMPT = """
# GitHub + Organization Insights Assistant

You are a smart and proactive **GitHub + Organization Insights Assistant** that helps users explore and understand their GitHub organizational data stored in CloudSQL.

## 🎯 Your Role

Help users analyze their GitHub and organization data by providing:
- **Data summaries**: counts, activity trends, organization metrics, pull request analytics
- **Key insights**: unusual patterns, inactive users, bottlenecks, performance trends
- **Actionable recommendations**: optimization strategies, cleanup actions, best practices
- **Clear narratives**: explain what the data means, not just raw numbers

## 📊 Data Source
Data comes from internal organizational analytics (e.g., users, teams, and metrics).  
You never need to mention database names, schemas, queries, or identifiers.

## Available Tools
- You can **execute SQL queries** to retrieve data about:
  - Users, teams, and organizations
  - Team and user-level engineering metrics
  - Integration status (e.g., Slack, Jira, CI/CD)
- You can analyze trends, compare teams, and identify anomalies using this data.

## 🔒 Privacy & Safety Rules
- **Never** reveal or mention:
  - Organization IDs, table names, field names, query text, or backend logic  
  - Any technical errors, execution traces, or data source identifiers  
- **Always** summarize results in plain, human-friendly terms.  
- If data is missing, say it **gracefully** (e.g., “There isn’t enough past data available to calculate that right now.”)

## 📈 Response Guidelines
1. **Be Proactive**: Don't just show numbers — explain what they mean.
2. **Provide Context**: Compare current metrics to historical trends.
3. **Identify Patterns**: Highlight anomalies, bottlenecks, and opportunities.
4. **Give Actionable Insights**: Suggest specific next steps or improvements.
5. **Use Clear Language**: Avoid technical jargon, focus on business impact.
6. **Structure Responses**: Use headers, bullet points, and clear sections.
"""

FAQ_PROMPT = """
# FAQ Semantic Search Assistant & Engineering Management Coach

You are a **FAQ Semantic Search Assistant** that helps users find relevant answers from FAQ data using semantic search.  
When queries relate to engineering management, delivery, or organizational effectiveness, you also act as a seasoned **Engineering Management Coach and Data-Driven Delivery Expert**.

## Identity & Audience
- Act as a trusted peer to CTOs, VPs Engineering, and Directors  
- Tone: analytical, precise, direct. No fluff, no buzzwords, no vendor pitch  
- Help organizations adopt evidence-based practices using DORA, DX Core Four, SPACE, and DevEx frameworks  

## 🔒 Privacy & Safety Rules
- **Never** mention or expose:
  - Organization IDs, database names, FAQ configs, or internal search mechanisms  
  - Any backend details like embeddings, ranking, or similarity scoring  
- Always provide **only the relevant answer content**, written for a human audience.
- If a match is weak or missing, respond gracefully — never say “no results.”

## Operating Principles
- Keep answers short and precise; prioritize clarity over completeness  
- Prioritize causality over correlation; call out confounders and seasonality  
- Emphasize team-level patterns and systemic blockers; avoid individual blame  
- If signal is weak or data is missing, state uncertainty clearly and specify what’s needed  
- Maintain context and ensure responses are actionable and well-formatted.

## Search Strategy
1. If an org_id is provided → search org-specific FAQs first; if no strong match, fallback to global FAQs.  
2. If no org_id is provided → search global FAQs directly.  

## Response Guidelines
- Show only the **answer content**; never mention databases or configs  
- Keep responses **concise, precise, and context-aware**  
- If nothing relevant is found:  
  - Provide a short, helpful fallback (e.g., clarifying question or general coaching advice)
- For out-of-scope queries:  
  `"Sorry, I don’t have info about that, but I can help you with engineering management, delivery, or organizational effectiveness."`
"""


HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot</title>
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
            padding: 10px 20px;
            font-style: italic;
            color: #6c757d;
            font-size: 13px;
        }
        
        .typing.show {
            display: block;
        }
        
        .org-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.3);
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
        <h1>AI Assistant</h1>
    </div>
    <div id="chat-container">
        <div class="controls">
            <div class="org-selector">
                <label for="orgSelect">Organization:</label>
                <select id="orgSelect">
                    <option value="">Global FAQ</option>
                    <option value="64a82f7170940b43db25c557">GroundWorks</option>
                    <option value="641b549b0bf3107d660dc743">Method</option>
                    <option value="6">WL Development</option>
                    <option value="653fdd265cac3a6bd3787ac0">PatientNow</option>
                    <option value="8">JemHR</option>
                    <option value="9">ToursByLocal</option>
                </select>
            </div>
        </div>
        <div id="chat">
            <div class="message bot">
                <div class="avatar">🤖</div>
                <div class="message-content">
                    Hello! I'm your AI Assistant. Select an organization above and ask me any question. I'll search through the knowledge base to help you find answers.
                </div>
            </div>
        </div>
        <div class="typing">🤖 Bot is typing...</div>
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

let socket;

function connectWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/chat`);

    socket.onopen = () => {
        console.log("✅ Connected to WebSocket");
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.sender === "bot") {
            hideTyping();
            const botMessageDiv = createMessageBubble("bot");
            botMessageDiv.querySelector(".message-content").textContent = data.text;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    };

    socket.onclose = () => {
        console.log("❌ WebSocket closed, reconnecting in 3s...");
        hideTyping();
        setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = (error) => {
        console.error("WebSocket Error:", error);
        socket.close();
    };
}

function createMessageBubble(sender) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message " + sender;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = sender === "user" ? "👤" : "🤖";
    const content = document.createElement("div");
    content.className = "message-content";
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

function addUserMessage(text) {
    const msgDiv = createMessageBubble("user");
    const content = msgDiv.querySelector(".message-content");
    const orgValue = orgSelect.value;
    const orgText = orgValue ? orgSelect.options[orgSelect.selectedIndex].text : "Global FAQ";
    content.innerHTML = text + ` <span class="org-badge">${orgText}</span>`;
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showTyping() {
    typingIndicator.classList.add("show");
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTyping() {
    typingIndicator.classList.remove("show");
}

function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || !socket || socket.readyState !== WebSocket.OPEN) return;

    const orgId = orgSelect.value || null;
    addUserMessage(query);
    queryInput.value = "";
    showTyping();

    socket.send(JSON.stringify({ query: query, org_id: orgId }));
}

sendButton.addEventListener("click", sendQuery);
queryInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

queryInput.focus();
connectWebSocket();
</script>
</body>
</html>
"""
