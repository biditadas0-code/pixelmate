from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
import os
import uuid
from dotenv import load_dotenv
from pymongo import MongoClient

app = Flask(__name__)
load_dotenv()

app.secret_key = os.environ.get("SECRET_KEY") or "pixelmate-local-session"

MONGO_URI = os.environ.get("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chatbot_db"]
messages_collection = db["messages"]

# ---- GROQ API SETUP ----

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Please set your Groq API key."
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ---- CHATBOT PERSONALITY ----

SYSTEM_PROMPT = (
    "You are PixelMate, a premium AI assistant for software developers. "
    "You are not ChatGPT, GPT-4, or any OpenAI product, and you must never claim to be one. "
    "If asked who made you or what model you are, say: "
    "I'm PixelMate, a custom chatbot built by Bidita using the Groq API. "
    "Help with coding, debugging, code review, project explanations, and security suggestions. "
    "Be precise, practical, and structured. Prefer working code over vague advice. "
    "When you show code, use fenced Markdown with a language tag. "
    "Call out assumptions, edge cases, and what to test next. "
    "If information is missing, ask a short clarifying question, then still give a useful first pass."
)

MODE_HINTS = {
    "debug": (
        "Mode: Bug Detective. Identify the likely root cause, explain it in plain language, "
        "then provide a minimal corrected snippet. Separate diagnosis from the fix."
    ),
    "review": (
        "Mode: Code Review. Review like a senior engineer. Group findings by severity "
        "(blocker, issue, nit). Mention what is already solid, then suggest concrete patches."
    ),
    "project": (
        "Mode: Project Brain. Explain the project from the pasted files or notes: structure, "
        "how pieces connect, risks, and the next change to make. File upload is not available."
    ),
    "security": (
        "Mode: Security Scan. This is advisory, not an automated pentest. List likely risks, "
        "why they matter, and safer alternatives. Do not provide exploit steps."
    ),
}

# Per-browser conversation memory. Resets when the Flask process restarts.
conversations = {}


def system_messages(mode):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    hint = MODE_HINTS.get(mode)
    if hint:
        messages.append({"role": "system", "content": hint})
    return messages


def get_chat_id(reset=False):
    if reset or "chat_id" not in session:
        session["chat_id"] = str(uuid.uuid4())
    return session["chat_id"]


def get_history(mode=None, reset=False):
    chat_id = get_chat_id(reset=reset)
    if reset or chat_id not in conversations:
        conversations[chat_id] = system_messages(mode)
        return conversations[chat_id]

    rest = [item for item in conversations[chat_id] if item.get("role") != "system"]
    conversations[chat_id] = system_messages(mode) + rest
    return conversations[chat_id]


def trim_history(history):
    system = [item for item in history if item.get("role") == "system"]
    rest = [item for item in history if item.get("role") != "system"]
    return system + rest[-10:]


def looks_like_openai_identity(text):
    lower = (text or "").lower()
    claims = (
        "i am chatgpt",
        "i'm chatgpt",
        "i am gpt-4",
        "i'm gpt-4",
        "i am an openai",
        "i'm an openai",
    )
    return any(claim in lower for claim in claims)


# ---- HOME PAGE ----

@app.route("/")
def home():
    return render_template("index.html")


# ---- CHAT API ----

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")
    reset = bool(data.get("reset"))
    mode = data.get("mode")
    if mode not in MODE_HINTS:
        mode = None

    history = get_history(mode=mode, reset=reset)

    if not str(user_message).strip():
        if reset:
            return jsonify({
                "reply": "New chat ready."
            })
        return jsonify({
            "reply": "Please type something!"
        })

    history.append({
        "role": "user",
        "content": user_message
    })
    conversations[session["chat_id"]] = trim_history(history)
    history = conversations[session["chat_id"]]

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=history,
            temperature=0.4,
            max_tokens=2048
        )

        reply = response.choices[0].message.content or ""
        if looks_like_openai_identity(reply):
            reply = (
                "I'm PixelMate, a custom chatbot built by Bidita using the Groq API — not ChatGPT!"
            )

        try:
            messages_collection.insert_one({
                "user_message": user_message,
                "bot_reply": reply,
                "mode": mode or "general",
                "chat_id": session.get("chat_id"),
            })
        except Exception as db_error:
            print("Mongo save skipped:", str(db_error))

        history.append({
            "role": "assistant",
            "content": reply
        })
        conversations[session["chat_id"]] = trim_history(history)

        return jsonify({
            "reply": reply
        })

    except Exception as e:

        if history and history[-1].get("role") == "user":
            history.pop()
            conversations[session["chat_id"]] = history

        print("Groq API Error:", str(e))

        return jsonify({
            "reply": "Sorry, I couldn't connect to the AI right now."
        }), 500


# ---- START FLASK ----

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000
    )
