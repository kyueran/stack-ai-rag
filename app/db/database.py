import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.core.config import Settings
from app.db.schema import SCHEMA_STATEMENTS


class Database:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def build_database(settings: Settings) -> Database:
    database_path = settings.data_dir / "indexes" / "rag.sqlite"
    return Database(database_path)
