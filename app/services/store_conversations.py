import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings


settings = get_settings()

# token usage columns, added to databases created before token tracking existed
TOKEN_COLUMNS = {
    "input_tokens": "INTEGER NOT NULL DEFAULT 0",
    "output_tokens": "INTEGER NOT NULL DEFAULT 0",
    "llm_calls": "INTEGER NOT NULL DEFAULT 0",
    "token_usage_json": "TEXT NOT NULL DEFAULT '[]'",
}


# add any missing column to an already existing table
def migrate_schema(con: sqlite3.Connection) -> None:
    existing = {row[1] for row in con.execute("PRAGMA table_info(conversation_audit)")}
    for column, definition in TOKEN_COLUMNS.items():
        if column not in existing:
            con.execute(f"ALTER TABLE conversation_audit ADD COLUMN {column} {definition}")


# initialize the sqlite database
def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS conversation_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            llm_calls INTEGER NOT NULL DEFAULT 0,
            token_usage_json TEXT NOT NULL DEFAULT '[]'
        )"""
    )
    migrate_schema(con)
    con.commit()
    con.close()


# update the conversation_audit table
def add_conversation_audit(
    question: str,
    source_used: str,
    trace: list[str],
    token_usage: dict | None = None,
) -> None:
    usage = token_usage or {}
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """INSERT INTO conversation_audit(
            created_at, question, source_used, trace_json,
            input_tokens, output_tokens, llm_calls, token_usage_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            source_used,
            json.dumps(trace),
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            int(usage.get("llm_calls", 0) or 0),
            json.dumps(usage.get("by_node", [])),
        ),
    )
    con.commit()
    con.close()


# aggregate token spend across stored conversations
def get_token_usage_totals() -> dict:
    con = sqlite3.connect(settings.audit_db_path)
    row = con.execute(
        """SELECT COUNT(*), COALESCE(SUM(input_tokens), 0),
                  COALESCE(SUM(output_tokens), 0), COALESCE(SUM(llm_calls), 0)
           FROM conversation_audit"""
    ).fetchone()
    con.close()
    conversations, input_tokens, output_tokens, llm_calls = row
    return {
        "conversations": conversations,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "llm_calls": llm_calls,
    }