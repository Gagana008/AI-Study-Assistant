# 🎓 AI Study Assistant

A fully local, AI-powered study companion. Upload your PDF study materials
and get an AI chatbot, auto-generated summaries, key points, and practice
quizzes — all powered by a local **Ollama** installation. No OpenAI key, no
Pinecone account, no paid API of any kind is required.

## Features
1. Upload one or more PDF study materials.
2. Automatic text extraction and chunking of the documents.
3. Local vector embeddings via Ollama's `nomic-embed-text` model.
4. Fast semantic search over your material using **FAISS**.
5. AI chatbot (RAG) that answers questions using your uploaded material.
6. One-click AI-generated **summary** of the material.
7. One-click AI-generated **key points** list.
8. AI-generated **multiple-choice quiz** from your material.
9. Instant quiz scoring with correct-answer review.
10. Quiz history and scores saved permanently in a local **SQLite** database.
11. Simple, student-friendly **dashboard** (documents uploaded, quizzes taken, average score, best score, recent history).
12. Modern, responsive, sidebar-navigation UI.
13. Clear error handling and loading indicators throughout.

## Technologies Used
- Python 3.12
- Flask
- Ollama — local LLM inference
  - `llama3.2` for chat, summaries, key points, and quiz generation
  - `nomic-embed-text` for embeddings
- FAISS (`faiss-cpu`) for vector similarity search
- pypdf for PDF text extraction
- SQLite (built into Python) for quiz history storage
- HTML5 / CSS3 / vanilla JavaScript

## Folder Structure
```
04_ai_study_assistant/
│
├── app.py                 # Main Flask application (all routes)
├── database.py             # SQLite setup, quiz history, dashboard stats
├── vector_store.py          # FAISS vector store + Ollama embeddings
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── uploads/                # Temporary storage for uploaded PDFs (auto-cleared)
│   └── .gitkeep
│
├── data/                    # SQLite database file lives here
│   └── .gitkeep
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## Prerequisites
- **Python 3.12** (recommended; the packages in `requirements.txt` are
  chosen for Python 3.12 compatibility on Windows, macOS, and Linux).
- **Ollama** installed and running locally: https://ollama.com/download

## Installation (Windows PowerShell)

1. **Open PowerShell in the project folder**:
   ```powershell
   cd 04_ai_study_assistant
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   ```

3. **Activate it**:
   ```powershell
   venv\Scripts\Activate.ps1
   ```
   > If you get an execution-policy error, run PowerShell as Administrator and
   > execute: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Pull the required Ollama models** (skip any you already have installed):
   ```powershell
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
   Make sure Ollama is running — it typically runs automatically in the
   background after installation. If not, run `ollama serve` in a separate
   terminal window.

6. **(Optional) Configure environment variables**:
   ```powershell
   copy .env.example .env
   ```
   Edit `.env` if you want to change the model names, Ollama host, or the
   Flask secret key.

## Running the Application
```powershell
python app.py
```
Then open your browser at: **http://localhost:5004**

## macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
python app.py
```

## How to Use
1. Open the app and go to the **Upload Material** tab.
2. Upload one or more PDF study documents.
3. Go to **Ask Questions** to chat with an AI about your material.
4. Go to **Summary** to generate a quick review summary.
5. Go to **Key Points** to get an extracted bullet list of important concepts.
6. Go to **Quiz Me**, choose how many questions you want, and generate a quiz.
7. Answer the quiz and click **Submit Quiz** to see your score.
8. Check the **Dashboard** tab to see your quiz history and stats over time.

## Database
The SQLite database file (`data/study_assistant.db`) is created
automatically the first time you run the app — no manual setup needed. It
stores:
- `documents` — metadata about uploaded files.
- `quiz_history` — every quiz attempt with score and timestamp.
- `quiz_questions` — the individual questions/answers for each attempt.

## Troubleshooting
| Problem | Solution |
|---|---|
| `Could not connect to Ollama` | Make sure Ollama is installed and running. Run `ollama serve` in a terminal, then retry. |
| Errors mentioning `nomic-embed-text` | Run `ollama pull nomic-embed-text` to install the embedding model. |
| Errors mentioning `llama3.2` | Run `ollama pull llama3.2` to install the chat/generation model. |
| Quiz or summary generation is slow | Large PDFs and larger models take longer. Try a smaller PDF, fewer questions, or a smaller model. |
| "Could not extract text from ..." | The PDF is likely a scanned image without selectable text. Use a text-based PDF, or OCR it first. |
| Port 5004 already in use | Edit the last line of `app.py` and change `port=5004` to a free port. |
| `faiss-cpu` fails to install | Make sure you're using Python 3.12 (not 3.13/3.14) and the latest `pip` (`python -m pip install --upgrade pip`). |
| Quiz history not showing up | Make sure the `data/` folder is writable; the app creates `study_assistant.db` there automatically. |

## Notes
- Everything runs 100% locally through Ollama — no API keys, no cloud costs.
- Uploaded PDFs are deleted from the server immediately after their text is extracted; only the extracted text (as vector embeddings) is kept in memory for the current server session.
- The vector index is in-memory and rebuilt each time you restart the server — re-upload your material after a restart if you want to keep asking questions or generating quizzes about it. Quiz history in SQLite, however, is permanent.
