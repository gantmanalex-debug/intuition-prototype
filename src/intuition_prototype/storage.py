import sqlite3
import json
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from intuition_prototype.records import (
    Action, ActionKind, Episode, Hypothesis, Intervention, InterventionRecord,
    Metrics, Observation, Prediction, Question, Record, Result,
)


def initialize(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS demo_entries "
            "(id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS inquiry_episodes ("
            "id TEXT PRIMARY KEY, scenario TEXT NOT NULL, seed INTEGER NOT NULL, "
            "budget INTEGER NOT NULL, cost INTEGER NOT NULL, "
            "stop_reason TEXT NOT NULL, interpretation TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS inquiry_records ("
            "episode_id TEXT NOT NULL REFERENCES inquiry_episodes(id), "
            "position INTEGER NOT NULL, record_id TEXT NOT NULL, "
            "kind TEXT NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY (episode_id, position), UNIQUE (episode_id, record_id))"
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


@dataclass(frozen=True)
class SavedEpisode:
    id: str
    scenario: str
    seed: int
    episode: Episode


_RECORD_TYPES = {
    record_type.__name__: record_type for record_type in (
        Observation, Hypothesis, Prediction, Question, InterventionRecord, Result,
    )
}


def _decode_record(kind: str, payload: str) -> Record:
    if kind not in _RECORD_TYPES:
        raise ValueError(f"Unknown persisted record kind: {kind}")
    fields = json.loads(payload)
    if kind == "Observation":
        fields["metrics"] = Metrics(**fields["metrics"])
    elif kind == "Hypothesis":
        fields["evidence_ids"] = tuple(fields["evidence_ids"])
    elif kind == "InterventionRecord":
        action = fields["action"]
        fields["action"] = Action(
            ActionKind(action["kind"]),
            None if action["intervention"] is None else Intervention(action["intervention"]),
        )
    return _RECORD_TYPES[kind](**fields)


def validate_episode(episode: Episode) -> None:
    if not 0 <= episode.cost <= episode.budget <= 20 or len(episode.records) > 100:
        raise ValueError("Episode violates cost, budget, or record bounds.")
    if episode.stop_reason not in (
        "budget_exhausted", "supported_interpretation", "investigations_exhausted",
    ):
        raise ValueError("Unknown episode stop reason.")
    seen: dict[str, Record] = {}
    cost = 0
    for record in episode.records:
        if type(record).__name__ not in _RECORD_TYPES or record.id in seen:
            raise ValueError("Invalid or duplicate record.")
        references: list[tuple[str, type | tuple[type, ...]]] = []
        if isinstance(record, Hypothesis):
            references.extend((key, (Observation, Result)) for key in record.evidence_ids)
        elif isinstance(record, Prediction):
            references.append((record.hypothesis_id, Hypothesis))
        elif isinstance(record, InterventionRecord):
            references.append((record.question_id, Question))
            if record.prediction_id is not None:
                references.append((record.prediction_id, Prediction))
            if record.cost != record.action.cost:
                raise ValueError("Recorded action cost does not match action.")
            cost += record.cost
        elif isinstance(record, Result):
            references.extend((
                (record.intervention_id, InterventionRecord),
                (record.observation_id, Observation),
                (record.prediction_id, Prediction),
            ))
        for key, expected in references:
            if key not in seen or not isinstance(seen[key], expected):
                raise ValueError(f"Invalid or forward evidence/claim reference: {key}")
        if isinstance(record, Result):
            action = seen[record.intervention_id]
            prediction = seen[record.prediction_id]
            observation = seen[record.observation_id]
            if (
                not isinstance(action, InterventionRecord)
                or action.action.kind != ActionKind.TEST
                or action.prediction_id != record.prediction_id
                or not isinstance(prediction, Prediction)
                or not isinstance(observation, Observation)
                or record.matched != prediction.matches(observation.metrics)
            ):
                raise ValueError("Result contradicts its linked action, prediction, or evidence.")
        seen[record.id] = record
    if cost != episode.cost:
        raise ValueError("Episode cost does not equal the sum of its actions.")


def save_episode(database_path: Path, scenario: str, seed: int, episode: Episode) -> str:
    validate_episode(episode)
    initialize(database_path)
    episode_id = uuid4().hex
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO inquiry_episodes "
            "(id, scenario, seed, budget, cost, stop_reason, interpretation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id, scenario, seed, episode.budget, episode.cost,
                episode.stop_reason, episode.interpretation,
            ),
        )
        connection.executemany(
            "INSERT INTO inquiry_records VALUES (?, ?, ?, ?, ?)",
            [
                (episode_id, position, record.id, type(record).__name__, json.dumps(asdict(record)))
                for position, record in enumerate(episode.records)
            ],
        )
    return episode_id


def load_episode(database_path: Path, episode_id: str) -> SavedEpisode:
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT scenario, seed, budget, cost, stop_reason, interpretation "
            "FROM inquiry_episodes WHERE id = ?", (episode_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"Episode not found: {episode_id}")
        records = connection.execute(
            "SELECT kind, payload FROM inquiry_records WHERE episode_id = ? ORDER BY position",
            (episode_id,),
        ).fetchall()
    episode = Episode(
        tuple(_decode_record(kind, payload) for kind, payload in records),
        row[2], row[3], row[4], row[5],
    )
    validate_episode(episode)
    return SavedEpisode(episode_id, row[0], row[1], episode)
