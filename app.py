import os, time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(
    http_options=types.HttpOptions(timeout=60000))

MODELS = ["gemini-3.5-flash-lite",
          "gemini-3.1-flash-lite",
          "gemini-3.8-flash"]

SYSTEM_PROMPT = """You are an AI Interview Coach.
Ask the user for their target job role, then ask
ONE interview question at a time. Pick a random topic
each time. Never repeat a question.
After each answer, give short feedback, a score out
of 10, and then ask the next question.
Keep replies short and friendly."""

app = Flask(__name__)
sessions = {}

def ask(sid, text):
    history = sessions.setdefault(sid, [])
    history.append(types.Content(
        role="user", parts=[types.Part(text=text)]))
    for model in MODELS:
        for _ in range(2):
            try:
                r = client.models.generate_content(
                    model=model, contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=1.0))
                history.append(types.Content(
                    role="model",
                    parts=[types.Part(text=r.text)]))
                return r.text
            except Exception:
                time.sleep(2)
    history.pop()
    return "Server is busy. Please send again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json()
    return jsonify({"reply": ask(d["session"], d["message"])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    