"""
AI Study Assistant
-------------------
A local, offline-friendly study companion built with Flask + Ollama.

Pipeline:
    PDF upload -> text extraction -> chunking -> embeddings (nomic-embed-text)
    -> FAISS vector index -> retrieval -> llama3.2 for chat answers,
    summaries, key points and quiz generation.

Also includes:
- A chatbot for asking questions about the uploaded material.
- AI-generated summary and key points.
- An AI-generated multiple-choice quiz with scoring.
- Quiz history saved in SQLite, shown on a simple dashboard.

No paid APIs or API keys are required - everything runs through a local
Ollama server.
"""

import os
import re
import json
import uuid
import requests
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from dotenv import load_dotenv

import database
from vector_store import store

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 MB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create the SQLite tables on startup if they don't already exist.
database.init_db()

# In-memory chat history keyed by browser session id.
chat_histories = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts).strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def call_ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama's /api/generate endpoint with a single prompt string."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4},
    }
    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running "
            f"(`ollama serve`) and the model is pulled (`ollama pull {model}`)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama returned an error: {exc}") from exc

    return response.json().get("response", "")


def call_ollama_chat(messages: list, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama's /api/chat endpoint with a list of role/content messages."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running "
            f"(`ollama serve`) and the model is pulled (`ollama pull {model}`)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama returned an error: {exc}") from exc

    return response.json().get("message", {}).get("content", "")


def extract_json_array(text: str):
    """Pull the first valid JSON array out of a raw LLM text response."""
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in the model output.")
    return json.loads(text[start:end + 1])


def get_session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
    return session["session_id"]


# ---------------------------------------------------------------------------
# Routes - pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    get_session_id()
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes - document upload
# ---------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    """Accept one or more PDFs, extract text, chunk, embed and index them."""
    files = request.files.getlist("pdf_files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files selected."}), 400

    processed_files = []
    total_chunks = 0

    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"error": f"'{file.filename}' is not a PDF file."}), 400

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

        try:
            file.save(filepath)
            text = extract_text_from_pdf(filepath)
        except Exception as exc:
            return jsonify({"error": f"Failed to process '{filename}': {exc}"}), 400
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

        if not text or len(text) < 30:
            return jsonify({
                "error": f"Could not extract text from '{filename}'. "
                         "It may be a scanned/image-only PDF."
            }), 400

        chunks = chunk_text(text)
        metadatas = [{"source": filename} for _ in chunks]

        try:
            store.add_texts(chunks, metadatas)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 502

        database.log_document(filename, len(chunks))
        total_chunks += len(chunks)
        processed_files.append(filename)

    return jsonify({
        "message": f"Indexed {total_chunks} chunks from {len(processed_files)} file(s).",
        "files": processed_files,
    })


@app.route("/clear_documents", methods=["POST"])
def clear_documents():
    store.clear()
    return jsonify({"message": "All indexed study material has been cleared."})


# ---------------------------------------------------------------------------
# Routes - chatbot (RAG)
# ---------------------------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please type a question."}), 400
    if store.is_empty():
        return jsonify({"error": "Please upload study material first."}), 400

    try:
        retrieved = store.similarity_search(question, k=TOP_K)
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['metadata'].get('source', 'document')}]\n{c['text']}"
            for c in retrieved
        ) or "(No relevant context found.)"

        system_prompt = (
            "You are a friendly study assistant. Answer the student's question "
            "using the CONTEXT from their uploaded study material as the primary "
            "source of truth. If the context does not contain the answer, say so "
            "clearly, then optionally give a brief general-knowledge answer "
            "labeled as such. Keep answers clear and exam-friendly."
        )
        user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}"

        answer = call_ollama_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502

    session_id = get_session_id()
    history = chat_histories.setdefault(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    sources = sorted({c["metadata"].get("source", "document") for c in retrieved})
    return jsonify({"answer": answer, "sources": sources})


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session_id = get_session_id()
    chat_histories[session_id] = []
    return jsonify({"message": "Chat history cleared."})


# ---------------------------------------------------------------------------
# Routes - summary & key points
# ---------------------------------------------------------------------------
@app.route("/summary", methods=["POST"])
def summary():
    if store.is_empty():
        return jsonify({"error": "Please upload study material first."}), 400

    text = store.get_all_text()
    prompt = f"""
You are a study assistant. Read the STUDY MATERIAL below and write a clear,
well-organized summary (around 150-250 words) that a student could use to
quickly review the topic before an exam.

STUDY MATERIAL:
{text}
"""
    try:
        result = call_ollama_generate(prompt.strip())
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"summary": result.strip()})


@app.route("/key_points", methods=["POST"])
def key_points():
    if store.is_empty():
        return jsonify({"error": "Please upload study material first."}), 400

    text = store.get_all_text()
    prompt = f"""
You are a study assistant. Read the STUDY MATERIAL below and extract the most
important key points a student must remember. Respond with ONLY a valid JSON
array of short strings (no markdown, no extra commentary), like:
["key point 1", "key point 2", "key point 3"]
Produce between 6 and 12 key points.

STUDY MATERIAL:
{text}
"""
    try:
        raw = call_ollama_generate(prompt.strip())
        points = extract_json_array(raw)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except (ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"The AI model returned an unexpected format: {exc}"}), 502

    points = [p for p in points if isinstance(p, str) and p.strip()]
    return jsonify({"key_points": points})


# ---------------------------------------------------------------------------
# Routes - quiz
# ---------------------------------------------------------------------------
@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    if store.is_empty():
        return jsonify({"error": "Please upload study material first."}), 400

    data = request.get_json(silent=True) or {}
    try:
        num_questions = int(data.get("num_questions", 5))
        num_questions = max(1, min(num_questions, 15))
    except (ValueError, TypeError):
        num_questions = 5

    text = store.get_all_text()
    prompt = f"""
You are an expert exam question writer. Read the STUDY MATERIAL below and
create exactly {num_questions} multiple-choice questions (MCQs) that test a
student's understanding of it.

Rules:
- Each question must have exactly 4 answer options.
- Exactly one option must be correct.
- Base every question strictly on the study material.
- Respond with ONLY a valid JSON array, no explanations, no markdown.
- Use this exact structure:
[
  {{
    "question": "string",
    "options": ["option A", "option B", "option C", "option D"],
    "correct_answer": "the exact text of the correct option"
  }}
]

STUDY MATERIAL:
{text}
"""
    try:
        raw = call_ollama_generate(prompt.strip())
        questions = extract_json_array(raw)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except (ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"The AI model returned an unexpected format: {exc}"}), 502

    clean_questions = []
    for q in questions:
        if (
            isinstance(q, dict) and "question" in q and "options" in q
            and "correct_answer" in q and isinstance(q["options"], list)
            and len(q["options"]) == 4
        ):
            clean_questions.append({
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
            })

    if not clean_questions:
        return jsonify({"error": "The AI model did not return any valid questions. Try again."}), 502

    quiz_id = uuid.uuid4().hex
    session[f"quiz_{quiz_id}"] = clean_questions

    public_questions = [{"question": q["question"], "options": q["options"]} for q in clean_questions]
    return jsonify({"quiz_id": quiz_id, "questions": public_questions})


@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    data = request.get_json(silent=True) or {}
    quiz_id = data.get("quiz_id")
    answers = data.get("answers", {})
    topic = data.get("topic", "Study Material")

    quiz = session.get(f"quiz_{quiz_id}")
    if not quiz:
        return jsonify({"error": "Quiz not found or session expired."}), 404

    results = []
    score = 0
    for idx, q in enumerate(quiz):
        selected = answers.get(str(idx), "")
        is_correct = selected.strip() == q["correct_answer"].strip()
        if is_correct:
            score += 1
        results.append({
            "question": q["question"],
            "selected": selected,
            "correct_answer": q["correct_answer"],
            "is_correct": is_correct,
        })

    session.pop(f"quiz_{quiz_id}", None)

    database.save_quiz_attempt(topic, score, len(quiz), results)

    return jsonify({"score": score, "total": len(quiz), "results": results})


# ---------------------------------------------------------------------------
# Routes - dashboard
# ---------------------------------------------------------------------------
@app.route("/dashboard", methods=["GET"])
def dashboard():
    stats = database.get_dashboard_stats()
    history = database.get_quiz_history(limit=10)
    return jsonify({"stats": stats, "history": history})


if __name__ == "__main__":
    # Runs on port 5004 so it can run alongside other local projects.
    app.run(debug=True, port=5004)
