"""
database.py
-----------
SQLite persistence layer for the AI Study Assistant.

Stores:
- documents      : uploaded study materials (metadata only)
- quiz_history    : every quiz attempt, with score and timestamp
- quiz_questions   : the individual questions/answers for each attempt
                     (useful for reviewing past quizzes)

The database file is created automatically on first run inside the
`data/` folder, so there is nothing to set up manually.
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "study_assistant.db")


def get_connection():
    """Return a new SQLite connection with foreign keys enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all required tables if they do not already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            num_chunks INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            taken_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            selected_answer TEXT,
            correct_answer TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            FOREIGN KEY (quiz_id) REFERENCES quiz_history (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def log_document(filename: str, num_chunks: int):
    conn = get_connection()
    conn.execute(
        "INSERT INTO documents (filename, num_chunks, uploaded_at) VALUES (?, ?, ?)",
        (filename, num_chunks, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def save_quiz_attempt(topic: str, score: int, total: int, results: list) -> int:
    """
    Save a completed quiz attempt and its per-question results.
    `results` is a list of dicts: question, selected, correct_answer, is_correct.
    Returns the new quiz_history row id.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO quiz_history (topic, score, total, taken_at) VALUES (?, ?, ?, ?)",
        (topic, score, total, datetime.utcnow().isoformat()),
    )
    quiz_id = cur.lastrowid

    for r in results:
        cur.execute(
            """INSERT INTO quiz_questions
               (quiz_id, question, selected_answer, correct_answer, is_correct)
               VALUES (?, ?, ?, ?, ?)""",
            (quiz_id, r["question"], r.get("selected", ""), r["correct_answer"],
             1 if r["is_correct"] else 0),
        )

    conn.commit()
    conn.close()
    return quiz_id


def get_quiz_history(limit: int = 20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM quiz_history ORDER BY taken_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_stats():
    """Return simple aggregate stats for the student dashboard."""
    conn = get_connection()
    doc_count = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
    quiz_count = conn.execute("SELECT COUNT(*) AS c FROM quiz_history").fetchone()["c"]
    row = conn.execute(
        "SELECT AVG(score * 1.0 / total) AS avg_pct FROM quiz_history WHERE total > 0"
    ).fetchone()
    avg_pct = round((row["avg_pct"] or 0) * 100, 1)
    best = conn.execute(
        "SELECT score, total FROM quiz_history ORDER BY (score * 1.0 / total) DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return {
        "documents_uploaded": doc_count,
        "quizzes_taken": quiz_count,
        "average_score_percent": avg_pct,
        "best_score": f"{best['score']}/{best['total']}" if best else "N/A",
    }
