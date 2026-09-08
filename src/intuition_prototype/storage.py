import sqlite3
from contextlib import closing
from pathlib import Path


def initialize(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS demo_entries "
            "(id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
        )


def write_entry(database_path: Path, content: str) -> int:
    with closing(sqlite3.connect(database_path)) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO demo_entries (content) VALUES (?)", (content,)
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return an inserted row ID.")
        return cursor.lastrowid


def read_entry(database_path: Path, entry_id: int) -> str | None:
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT content FROM demo_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return None if row is None else row[0]


def count_entries(database_path: Path) -> int:
    with closing(sqlite3.connect(database_path)) as connection:
        return connection.execute("SELECT COUNT(*) FROM demo_entries").fetchone()[0]
