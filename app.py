import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in .env")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = (BASE_DIR / "chatbot_config").read_text(encoding="utf-8").strip()

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{SYSTEM_PROMPT}\n\nUser message:\n{message}",
        )
        return jsonify({"reply": response.text.strip()})
    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({"error": "Unable to get a response right now."}), 500


if __name__ == "__main__":
    app.run(debug=True)
