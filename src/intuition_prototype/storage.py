import sqlite3
import json
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from intuition_prototype.records import (
    Action, ActionKind, Episode, Hypothesis, Intervention, InterventionRecord,
    Metrics, Observation, Prediction, Question, Record, Result,
    AnswerRecord, CandidateScore, DecisionRecord, FuseRecord,
    LearnedCandidateScore,
)


def initialize(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
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
        columns = {row[1] for row in connection.execute("PRAGMA table_info(inquiry_episodes)")}
        if "policy" not in columns:
            connection.execute(
                "ALTER TABLE inquiry_episodes ADD COLUMN policy TEXT NOT NULL DEFAULT 'scripted'"
            )
        if "max_steps" not in columns:
            connection.execute(
                "ALTER TABLE inquiry_episodes ADD COLUMN max_steps INTEGER NOT NULL DEFAULT 12"
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
        DecisionRecord, FuseRecord, AnswerRecord,
    )
}


def _decode_record(kind: str, payload: str) -> Record:
    if kind not in _RECORD_TYPES:
        raise ValueError(f"Unknown persisted record kind: {kind}")
    fields = json.loads(payload)
    if kind == "Observation":
        fields["metrics"] = Metrics(**fields["metrics"])
    elif kind in ("Hypothesis", "DecisionRecord", "FuseRecord", "AnswerRecord"):
        fields["evidence_ids"] = tuple(fields["evidence_ids"])
    if kind == "InterventionRecord":
        fields["action"] = _decode_action(fields["action"])
    elif kind == "DecisionRecord":
        candidates = []
        for candidate in fields["candidates"]:
            candidate["action"] = _decode_action(candidate["action"])
            candidate["rejection_reasons"] = tuple(candidate["rejection_reasons"])
            if "model_sha256" in candidate:
                for key in ("feature_names", "standardized_features", "coefficients"):
                    candidate[key] = tuple(candidate[key])
                candidates.append(LearnedCandidateScore(**candidate))
            else:
                candidates.append(CandidateScore(**candidate))
        fields["candidates"] = tuple(candidates)
    elif kind == "FuseRecord":
        fields["triggers"] = tuple(fields["triggers"])
        fields["alternate_keys"] = tuple(fields["alternate_keys"])
    return _RECORD_TYPES[kind](**fields)


def _decode_action(fields: dict) -> Action:
    return Action(
        ActionKind(fields["kind"]),
        None if fields["intervention"] is None else Intervention(fields["intervention"]),
    )


def validate_episode(episode: Episode) -> None:
    if not 0 <= episode.cost <= episode.budget <= 20 or len(episode.records) > 100:
        raise ValueError("Episode violates cost, budget, or record bounds.")
    if episode.stop_reason not in (
        "budget_exhausted", "supported_interpretation", "investigations_exhausted",
        "inconclusive", "step_limit",
    ):
        raise ValueError("Unknown episode stop reason.")
    if episode.policy not in ("scripted", "heuristic", "learned"):
        raise ValueError("Unknown episode policy.")
    if type(episode.max_steps) is not int or not 1 <= episode.max_steps <= 12:
        raise ValueError("Invalid episode step limit.")
    seen: dict[str, Record] = {}
    model_fingerprints: set[str] = set()
    cost = steps = 0
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
            steps += 1
        elif isinstance(record, Result):
            references.extend((
                (record.intervention_id, InterventionRecord),
                (record.observation_id, Observation),
                (record.prediction_id, Prediction),
            ))
        elif isinstance(record, (DecisionRecord, FuseRecord)):
            references.extend((key, (Observation, Result)) for key in record.evidence_ids)
            if isinstance(record, DecisionRecord):
                references.append((record.hypothesis_id, Hypothesis))
                expected_type = LearnedCandidateScore if episode.policy == "learned" else CandidateScore
                if any(not isinstance(candidate, expected_type) for candidate in record.candidates):
                    raise ValueError("Candidate score schema contradicts the episode policy.")
                model_fingerprints.update(
                    candidate.model_sha256 for candidate in record.candidates
                    if isinstance(candidate, LearnedCandidateScore)
                )
                if len(model_fingerprints) > 1:
                    raise ValueError("A learned episode cannot mix model fingerprints.")
                keys = [candidate.key for candidate in record.candidates]
                if len(set(keys)) != len(keys) or not 0 <= record.remaining_budget <= episode.budget:
                    raise ValueError("Invalid decision candidates or remaining budget.")
                if record.chosen_key is not None:
                    eligible = [candidate for candidate in record.candidates if candidate.eligible]
                    winner = min(eligible, key=lambda item: (-item.score, item.key)) if eligible else None
                    if winner is None or winner.key != record.chosen_key:
                        raise ValueError("Recorded choice is not the highest eligible score.")
                if any(
                    candidate.eligible and candidate.cost + 1 > record.remaining_budget
                    for candidate in record.candidates
                ):
                    raise ValueError("Eligible candidate cannot afford its test and evidence update.")
            else:
                if record.disposition not in (
                    "answer", "switch_path", "budget_exhausted", "investigations_exhausted",
                    "inconclusive", "step_limit",
                ) or not 0 <= record.no_progress_count <= 3:
                    raise ValueError("Invalid fuse disposition or no-progress count.")
                if any(trigger not in (
                    "contradictory_prediction", "no_progress", "repeated_no_progress", "hard_step_bound",
                ) for trigger in record.triggers):
                    raise ValueError("Unknown fuse trigger.")
                if (
                    any(key not in {item.value for item in Intervention} for key in record.alternate_keys)
                    or len(set(record.alternate_keys)) != len(record.alternate_keys)
                    or (record.disposition == "switch_path" and not record.alternate_keys)
                ):
                    raise ValueError("Invalid fuse alternatives.")
        elif isinstance(record, AnswerRecord):
            references.append((record.action_id, InterventionRecord))
            references.extend((key, (Observation, Result, Hypothesis)) for key in record.evidence_ids)
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
        if isinstance(record, AnswerRecord):
            action = seen[record.action_id]
            if (
                not isinstance(action, InterventionRecord)
                or action.action.kind not in (ActionKind.ANSWER, ActionKind.STOP)
                or record.stop_reason != episode.stop_reason
                or record.claim != episode.interpretation
            ):
                raise ValueError("Terminal answer contradicts its action or episode.")
        seen[record.id] = record
    if cost != episode.cost:
        raise ValueError("Episode cost does not equal the sum of its actions.")
    if steps > episode.max_steps:
        raise ValueError("Episode exceeds its action-step limit.")


def save_episode(database_path: Path, scenario: str, seed: int, episode: Episode) -> str:
    validate_episode(episode)
    initialize(database_path)
    episode_id = uuid4().hex
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO inquiry_episodes "
            "(id, scenario, seed, budget, cost, stop_reason, interpretation, policy, max_steps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id, scenario, seed, episode.budget, episode.cost,
                episode.stop_reason, episode.interpretation, episode.policy, episode.max_steps,
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
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    initialize(database_path)
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT scenario, seed, budget, cost, stop_reason, interpretation, policy, max_steps "
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
        row[2], row[3], row[4], row[5], row[6], row[7],
    )
    validate_episode(episode)
    return SavedEpisode(episode_id, row[0], row[1], episode)
