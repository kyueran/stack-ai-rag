import sqlite3
from pathlib import Path

from app.db.database import Database


def test_database_initialization_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    db = Database(db_path)
    db.initialize()

    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

    assert {"documents", "chunks", "embeddings", "terms", "retrieval_logs"}.issubset(names)


def test_database_connection_enforces_foreign_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "rag.sqlite"
    db = Database(db_path)
    db.initialize()

    with db.connection() as conn:
        pragma_value = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert pragma_value == 1
