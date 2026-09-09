"""Context-first inquiry with separate goal and compatible-world state."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from intuition_prototype.benchmark_protocol import digest
from intuition_prototype import abstraction_protocol as protocol


MODEL_VERSION = "abstraction-context-preference-v2"
OPENINGS = {domain: protocol.PROTOCOL["domains"][domain]["opening"] for domain in protocol.DOMAINS}
QUESTIONS = protocol.QUESTIONS
QUESTION_ORDER = protocol.QUESTION_ORDER
MAX_QUESTIONS = protocol.MAX_TOTAL_QUESTIONS


def _history_rows(history, domain: str | None = None) -> tuple[tuple[str, str], ...]:
    if not isinstance(history, (tuple, list)):
        raise ValueError("History must be a sequence.")
    if domain is not None and domain not in protocol.DOMAINS:
        raise ValueError(f"Unknown inquiry domain: {domain!r}.")
    rows = []
    for item in history:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each history item must be a question-answer pair.")
        question, answer = item
        if question not in QUESTIONS or type(answer) is not str:
            raise ValueError("History contains an unknown question or non-string answer.")
        if domain is not None and question not in protocol.QUESTION_ORDER_BY_DOMAIN[domain]:
            raise ValueError("History question belongs to a different inquiry domain.")
        rows.append((question, answer))
    return tuple(rows)


def _groups(domain: str, leaves: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        group for group, members in protocol.GROUPS_BY_DOMAIN[domain].items()
        if any(leaf in members for leaf in leaves)
    )


def candidate_state(history=(), domain: str = "dog") -> dict:
    """Compatible sets from world/requirement answers only; context is ignored."""
    rows = _history_rows(history, domain)
    leaves = tuple(protocol.LEAVES_BY_DOMAIN[domain])
    physical_questions = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[domain]
    seen: set[str] = set()
    status_override = issue = None
    for question, answer in rows:
        if question == context_question:
            continue
        if question in seen:
            status_override = "inconsistent"
            issue = f"Repeated physical question {question!r} is illegal."
            leaves = ()
            break
        seen.add(question)
        if question not in physical_questions:
            raise ValueError("History contains a question outside this domain.")
        if answer not in protocol.PHYSICAL_ANSWERS:
            status_override = "out_of_model"
            issue = f"Physical answer token {answer!r} is outside the declared vocabulary."
            break
        if answer != "unknown":
            leaves = tuple(
                leaf for leaf in leaves
                if answer in protocol.allowed_answers(domain, leaf, question)
            )
        if not leaves:
            status_override = "inconsistent"
            issue = "No declared world matches all known answers."
            break
    groups = _groups(domain, leaves)
    status = status_override or ("unique" if len(leaves) == 1 else "ambiguous")
    return {
        "domain": domain,
        "status": status,
        "candidate_groups": groups,
        "candidate_leaves": leaves,
        "group_count": len(groups),
        "leaf_count": len(leaves),
        "uniform_group_mass": len(groups) / len(protocol.GROUPS_BY_DOMAIN[domain]),
        "uniform_leaf_mass": len(leaves) / len(protocol.LEAVES_BY_DOMAIN[domain]),
        "uniform_leaf_count_entropy_bits": math.log2(len(leaves)) if leaves else None,
        "answer": leaves[0] if status == "unique" else None,
        "supported": status == "unique",
        "issue": issue,
        "assumptions": (
            "Exact finite compatibility. Context is excluded; unknown evidence retains candidates. "
            "Mass and entropy use uniform candidate counts, not calibrated probabilities."
        ),
    }


def context_state(history=(), supplied_context: str | None = None, domain: str = "dog") -> dict:
    """Resolve only explicitly supplied or explicitly answered context."""
    rows = _history_rows(history, domain)
    context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[domain]
    if supplied_context is not None:
        data = protocol.CONTEXTS.get(supplied_context)
        if data is None or data["domain"] != domain:
            return {
                "domain": domain, "status": "out_of_model", "answer": None,
                "goal": None, "concern": None, "source": "explicit",
                "issue": "Supplied context is outside this domain's controlled vocabulary.",
            }
        return {
            "domain": domain, "status": "known", "answer": supplied_context,
            "goal": data["goal"], "concern": data["concern"],
            "source": "explicit supplied context", "issue": None,
        }
    answers = [answer for question, answer in rows if question == context_question]
    if not answers:
        return {
            "domain": domain, "status": "unknown", "answer": None,
            "goal": None, "concern": None, "source": "not supplied", "issue": None,
        }
    if len(answers) != 1:
        return {
            "domain": domain, "status": "inconsistent", "answer": None,
            "goal": None, "concern": None, "source": "clarification",
            "issue": "Context clarification was repeated.",
        }
    answer = answers[0]
    if answer == "unknown":
        return {
            "domain": domain, "status": "ambiguous", "answer": None,
            "goal": None, "concern": None, "source": "clarification",
            "issue": "The asker's goal or underlying requirement remains unknown.",
        }
    data = protocol.CONTEXTS.get(answer)
    if data is None or data["domain"] != domain:
        return {
            "domain": domain, "status": "out_of_model", "answer": None,
            "goal": None, "concern": None, "source": "clarification",
            "issue": "Clarification answer is outside this domain's vocabulary.",
        }
    return {
        "domain": domain, "status": "known", "answer": answer,
        "goal": data["goal"], "concern": data["concern"],
        "source": "asked clarification", "issue": None,
    }


def _binary_projection(world: dict, positive_groups: set[str], scope: str, goal: str, text: tuple[str, str]) -> dict:
    groups = set(world["candidate_groups"])
    if groups and groups <= positive_groups:
        value, answer = True, text[0]
    elif groups.isdisjoint(positive_groups):
        value, answer = False, text[1]
    else:
        return {
            "supported": False, "answer": None, "goal": goal, "scope": scope,
            "value": None, "retained_world_uncertainty": world["leaf_count"],
            "reason": "both projection values remain compatible",
        }
    return {
        "supported": True, "answer": answer, "goal": goal, "scope": scope,
        "value": value, "retained_world_uncertainty": world["leaf_count"],
        "reason": "the scoped projection is constant across compatible leaves",
    }


def goal_resolution(history=(), supplied_context: str | None = None, domain: str = "dog") -> dict:
    context = context_state(history, supplied_context, domain)
    world = candidate_state(history, domain)
    unresolved = {
        "supported": False, "answer": None, "goal": context["goal"],
        "scope": None, "value": None, "retained_world_uncertainty": world["leaf_count"],
    }
    if context["status"] != "known" or world["status"] in ("inconsistent", "out_of_model"):
        return {**unresolved, "reason": f"context={context['status']}, world={world['status']}"}
    goal = context["goal"]
    if goal == "honor_required_structure":
        return {
            "supported": True,
            "answer": (
                "The linked-list structure is explicitly required for implementation or education. "
                "Keep that requirement rather than substituting a different representation. "
                "This inquiry has established the requested scope only; it has not generated, "
                "implemented, or tested linked-list code."
            ),
            "goal": goal, "scope": "explicit_structure_requirement",
            "value": "linked_list_required",
            "retained_world_uncertainty": world["leaf_count"],
            "reason": "explicit supplied or clarified requirement, not inferred operation evidence",
        }
    if goal in ("explain_pattern", "recommend_representation"):
        scope = "fine_observed_pattern" if domain == "dog" else "fine_operation_profile"
        if world["status"] != "unique":
            return {**unresolved, "scope": scope, "reason": "multiple fine leaves remain"}
        answer = (
            f"Within the toy catalogue, the supported fine observed pattern is {world['answer']}."
            if domain == "dog"
            else (
                f"Within the toy requirements catalogue, {world['answer']} is the supported fine profile. "
                "Choose a representation from that profile rather than from the requested means alone."
            )
        )
        return {
            "supported": True, "answer": answer, "goal": goal, "scope": scope,
            "value": world["answer"], "retained_world_uncertainty": 1,
            "reason": "one compatible fine leaf",
        }
    if goal == "assess_outside":
        return _binary_projection(
            world, {"outside_directed"}, "outside_projection", goal,
            (
                "The observations support the outside-directed family within this toy catalogue.",
                "The observations do not support the outside-directed family within this toy catalogue.",
            ),
        )
    if goal == "assess_linked_fit":
        return _binary_projection(
            world, {"stable_identity", "splice_mutation"}, "node_linked_fit_projection", goal,
            (
                "The declared constraints support the node-linked family in this toy catalogue.",
                "The declared constraints do not select the node-linked family in this toy catalogue.",
            ),
        )
    if goal == "assess_indexing":
        return _binary_projection(
            world, {"indexed_collection"}, "indexed_access_projection", goal,
            (
                "The declared constraints support the indexed-access family in this toy catalogue.",
                "The declared constraints do not require the indexed-access family in this toy catalogue.",
            ),
        )
    if goal in ("characterize_change", "characterize_operations"):
        scope = "broad_observed_group" if domain == "dog" else "broad_operation_group"
        if world["group_count"] != 1:
            return {**unresolved, "scope": scope, "reason": "multiple broad groups remain"}
        group = world["candidate_groups"][0]
        answer = (
            f"The observations fit the broad {group} pattern; concern is context, not diagnosis or physical evidence."
            if domain == "dog"
            else f"The declared operations fit the broad {group} requirement family in this toy catalogue."
        )
        return {
            "supported": True, "answer": answer, "goal": goal, "scope": scope,
            "value": group, "retained_world_uncertainty": world["leaf_count"],
            "reason": "one compatible broad group",
        }
    return {**unresolved, "reason": "unsupported goal"}


def inquiry_state(history=(), supplied_context: str | None = None, domain: str = "dog") -> dict:
    return {
        "domain": domain,
        "context": context_state(history, supplied_context, domain),
        "world": candidate_state(history, domain),
        "goal_resolution": goal_resolution(history, supplied_context, domain),
    }


def legal_questions(history=(), supplied_context: str | None = None, domain: str = "dog") -> tuple[str, ...]:
    rows = _history_rows(history, domain)
    state = inquiry_state(rows, supplied_context, domain)
    context, world, goal = state["context"], state["world"], state["goal_resolution"]
    if (
        context["status"] in ("ambiguous", "inconsistent", "out_of_model")
        or world["status"] in ("inconsistent", "out_of_model")
        or goal["supported"]
    ):
        return ()
    asked = {question for question, _ in rows}
    context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[domain]
    if context["status"] == "unknown":
        return (context_question,) if context_question not in asked else ()
    physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    physical_count = sum(q in physical for q, _ in rows)
    total_limit = protocol.MAX_PHYSICAL_QUESTIONS if supplied_context is not None else protocol.MAX_TOTAL_QUESTIONS
    if physical_count >= protocol.MAX_PHYSICAL_QUESTIONS or len(rows) >= total_limit:
        return ()
    return tuple(q for q in physical if q not in asked)


def evidence_trace(history=(), supplied_context: str | None = None, domain: str = "dog") -> list[dict]:
    rows = _history_rows(history, domain)
    trace, prefix = [], ()
    before_world = candidate_state(prefix, domain)
    before_context = context_state(prefix, supplied_context, domain)
    context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[domain]
    for step, (question, answer) in enumerate(rows, 1):
        prefix += ((question, answer),)
        after_world = candidate_state(prefix, domain)
        after_context = context_state(prefix, supplied_context, domain)
        before_entropy = before_world["uniform_leaf_count_entropy_bits"]
        after_entropy = after_world["uniform_leaf_count_entropy_bits"]
        is_context = question == context_question
        trace.append({
            "evidence_id": f"evidence-{step}", "step": step, "question_id": question,
            "question": QUESTIONS[question], "observed_answer": answer,
            "kind": "context_clarification" if is_context else "world_observation",
            "world_evidence": not is_context,
            "context_before": before_context, "context_after": after_context,
            "before_groups": list(before_world["candidate_groups"]),
            "after_groups": list(after_world["candidate_groups"]),
            "before_leaves": list(before_world["candidate_leaves"]),
            "after_leaves": list(after_world["candidate_leaves"]),
            "group_count_reduction": before_world["group_count"] - after_world["group_count"],
            "leaf_count_reduction": before_world["leaf_count"] - after_world["leaf_count"],
            "uniform_count_entropy_reduction_bits": (
                before_entropy - after_entropy
                if before_entropy is not None and after_entropy is not None else None
            ),
            "interpretation": (
                "context only; compatible worlds unchanged" if is_context
                else "not observed; compatible worlds retained" if answer == "unknown"
                else "unsupported token; abstain" if answer not in protocol.PHYSICAL_ANSWERS
                else "exact elimination under declared compatibility"
            ),
        })
        before_world, before_context = after_world, after_context
    return trace


def expected_information_gain(history, question: str, supplied_context: str | None = None, domain: str = "dog") -> dict:
    """Nonlearned scoped count-information under explicit uniform assumptions."""
    rows = _history_rows(history, domain)
    world = candidate_state(rows, domain)
    context = context_state(rows, supplied_context, domain)
    physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    if (
        question not in physical or question in {q for q, _ in rows}
        or world["status"] in ("inconsistent", "out_of_model") or context["status"] != "known"
    ):
        raise ValueError("Information score requires known context and a legal physical question.")
    leaves = world["candidate_leaves"]
    prior_leaf = math.log2(len(leaves))
    prior_group = math.log2(max(1, world["group_count"]))
    before_projection = _projection_size(world, context["goal"])
    weighted_leaf = weighted_group = weighted_projection = goal_success = total = 0.0
    for answer in ("no", "yes"):
        weight = sum(
            1 / len(protocol.allowed_answers(domain, leaf, question))
            for leaf in leaves if answer in protocol.allowed_answers(domain, leaf, question)
        )
        if not weight:
            continue
        after_history = rows + ((question, answer),)
        after = candidate_state(after_history, domain)
        total += weight
        weighted_leaf += weight * math.log2(max(1, after["leaf_count"]))
        weighted_group += weight * math.log2(max(1, after["group_count"]))
        weighted_projection += weight * _projection_size(after, context["goal"])
        goal_success += weight * int(goal_resolution(after_history, supplied_context, domain)["supported"])
    if total:
        weighted_leaf /= total
        weighted_group /= total
        weighted_projection /= total
        goal_success /= total
    projection_reduction = before_projection - weighted_projection
    return {
        "leaf_entropy_reduction": prior_leaf - weighted_leaf,
        "group_entropy_reduction": prior_group - weighted_group,
        "projection_reduction": projection_reduction,
        "expected_goal_resolution": goal_success,
        "score": (
            8 * goal_success + 4 * projection_reduction
            + 2 * (prior_group - weighted_group) + prior_leaf - weighted_leaf
        ),
        "kind": "nonlearned_scoped_uniform_candidate_information",
        "assumption": "uniform compatible leaves and uniform allowed nuisance answers",
    }


def _projection_size(world: dict, goal: str | None) -> int:
    if goal in ("explain_pattern", "recommend_representation"):
        return world["leaf_count"]
    if goal in ("characterize_change", "characterize_operations"):
        return world["group_count"]
    positive = {
        "assess_outside": {"outside_directed"},
        "assess_linked_fit": {"stable_identity", "splice_mutation"},
        "assess_indexing": {"indexed_collection"},
    }.get(goal)
    if positive is None:
        return 0
    return len({group in positive for group in world["candidate_groups"]})


@dataclass(frozen=True)
class HiddenEpisode:
    episode_id: str
    physical_world_id: str
    domain: str
    group: str
    leaf: str
    answers: tuple[str, ...]
    split: str
    case_type: str
    context_variant: str
    context_explicit: bool
    context_answer: str
    goal: str | None
    concern: str | None

    def evaluator_record(self) -> dict:
        return {
            field: (list(value) if field == "answers" else value)
            for field, value in self.__dict__.items()
        }


class AbstractionQuestionAPI:
    """Lawful view: supplied context, observed history, compatible sets, legal questions."""

    __slots__ = ("__episode", "__history", "__supplied_context")

    def __init__(self, episode: HiddenEpisode):
        context_data = protocol.CONTEXTS.get(episode.context_answer)
        physical = (
            protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN.get(episode.domain, ())
        )
        if (
            episode.domain not in protocol.DOMAINS
            or episode.leaf not in protocol.LEAVES_BY_DOMAIN[episode.domain]
            or protocol.leaf_group(episode.domain, episode.leaf) != episode.group
            or len(episode.answers) != len(physical)
            or episode.case_type not in ("modelled", "physical_unknown", "inconsistent", "out_of_model")
            or type(episode.context_explicit) is not bool
            or (
                episode.context_answer != "unknown"
                and (
                    context_data is None or context_data["domain"] != episode.domain
                    or episode.goal != context_data["goal"]
                    or episode.concern != context_data["concern"]
                )
            )
            or (
                episode.context_answer == "unknown"
                and (episode.goal is not None or episode.concern is not None)
            )
            or (
                episode.case_type == "modelled"
                and any(
                    answer not in protocol.allowed_answers(episode.domain, episode.leaf, question)
                    for question, answer in zip(physical, episode.answers, strict=True)
                )
            )
        ):
            raise ValueError("Invalid hidden episode.")
        self.__episode = episode
        self.__history: tuple[tuple[str, str], ...] = ()
        self.__supplied_context = episode.context_answer if episode.context_explicit else None

    @property
    def domain(self) -> str:
        return self.__episode.domain

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        return self.__history

    @property
    def supplied_context(self) -> str | None:
        return self.__supplied_context

    def legal_questions(self) -> tuple[str, ...]:
        return legal_questions(self.__history, self.__supplied_context, self.domain)

    def observe(self) -> dict:
        physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[self.domain]
        return {
            "domain": self.domain,
            "opening": OPENINGS[self.domain],
            "supplied_context": self.__supplied_context,
            "history": [
                {
                    "evidence_id": f"evidence-{index}", "question_id": question,
                    "question": QUESTIONS[question], "answer": answer,
                }
                for index, (question, answer) in enumerate(self.__history, 1)
            ],
            "inquiry_state": inquiry_state(self.__history, self.__supplied_context, self.domain),
            "legal_questions": self.legal_questions(),
            "remaining_total_budget": (
                (protocol.MAX_PHYSICAL_QUESTIONS if self.__supplied_context is not None
                 else protocol.MAX_TOTAL_QUESTIONS) - len(self.__history)
            ),
            "remaining_physical_budget": (
                protocol.MAX_PHYSICAL_QUESTIONS - sum(q in physical for q, _ in self.__history)
            ),
        }

    def ask(self, question: str) -> dict:
        if question not in self.legal_questions():
            raise ValueError("Question is unknown, repeated, context-inappropriate, terminal, or over budget.")
        context_question = protocol.CONTEXT_QUESTION_BY_DOMAIN[self.domain]
        physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[self.domain]
        answer = (
            self.__episode.context_answer
            if question == context_question
            else self.__episode.answers[physical.index(question)]
        )
        self.__history += ((question, answer),)
        return self.observe()["history"][-1]


def full_state_key(history, supplied_context: str | None = None, domain: str = "dog") -> str:
    rows = _history_rows(history, domain)
    state = inquiry_state(rows, supplied_context, domain)
    return digest({
        "kind": "full_context_and_compatible_state", "domain": domain,
        "goal": state["context"]["goal"], "concern": state["context"]["concern"],
        "context_status": state["context"]["status"],
        "asked": sorted(question for question, _ in rows),
        "candidate_leaves": state["world"]["candidate_leaves"],
        "world_status": state["world"]["status"], "depth": len(rows),
    })


def memoryless_state_key(history, supplied_context: str | None = None, domain: str = "dog") -> str:
    rows = _history_rows(history, domain)
    state = inquiry_state(rows, supplied_context, domain)
    physical = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    latest = next(((q, a) for q, a in reversed(rows) if q in physical), None)
    return digest({
        "kind": "context_plus_latest_answer_ablation", "domain": domain,
        "goal": state["context"]["goal"], "concern": state["context"]["concern"],
        "context_status": state["context"]["status"],
        "asked": sorted(question for question, _ in rows),
        "latest_physical": latest, "world_status": state["world"]["status"], "depth": len(rows),
    })


@dataclass(frozen=True)
class PreferenceModel:
    version: str
    memoryless: bool
    tables: dict[str, dict[str, dict[str, float | int]]]
    state_counts: dict[str, int]
    provenance: dict
    model_sha256: str

    def document(self) -> dict:
        return {
            "version": self.version, "memoryless": self.memoryless, "tables": self.tables,
            "state_counts": self.state_counts, "provenance": self.provenance,
            "model_sha256": self.model_sha256,
        }

    def choose(self, history, legal: tuple[str, ...], supplied_context: str | None = None,
               domain: str = "dog") -> tuple[str | None, dict]:
        rows = _history_rows(history, domain)
        domain_order = protocol.QUESTION_ORDER_BY_DOMAIN[domain]
        if any(question not in domain_order for question in legal) or len(set(legal)) != len(legal):
            raise ValueError("Invalid legal-question set.")
        if not legal:
            return None, {"status": "no_legal_question", "scores": {}, "model_sha256": self.model_sha256}
        key = (
            memoryless_state_key(rows, supplied_context, domain)
            if self.memoryless else full_state_key(rows, supplied_context, domain)
        )
        if key not in self.tables or any(question not in self.tables[key] for question in legal):
            return None, {
                "status": "unsupported_state_abstention", "scores": {},
                "state_key": key, "model_sha256": self.model_sha256,
            }
        scores = {question: self.tables[key][question]["score"] for question in legal}
        chosen = min(legal, key=lambda q: (-scores[q], domain_order.index(q)))
        return chosen, {
            "status": "learned_count_ranking", "scores": scores,
            "score_interpretation": "smoothed optimal-label frequency; not calibrated probability",
            "state_key": key, "model_sha256": self.model_sha256,
        }


def fit_preference(examples: list[dict], provenance: dict, *, memoryless: bool = False) -> PreferenceModel:
    if not examples or any(row.get("split") != "train" for row in examples):
        raise ValueError("Preference fitting is training-only and requires nonempty examples.")
    required_provenance = {
        "fit_split", "protocol_sha256", "training_examples_sha256",
        "training_example_count", "training_episode_ids", "training_physical_world_ids", "label_rule",
    }
    if (
        set(provenance) != required_provenance or provenance["fit_split"] != "train"
        or provenance["protocol_sha256"] != protocol.PROTOCOL_SHA256
        or provenance["training_examples_sha256"] != digest(examples)
        or provenance["training_example_count"] != len(examples)
        or provenance["training_episode_ids"] != sorted({row["episode_id"] for row in examples})
        or provenance["training_physical_world_ids"]
        != sorted({row["physical_world_id"] for row in examples})
    ):
        raise ValueError("Invalid or incomplete training provenance.")
    expected_schema = {
        "episode_id", "physical_world_id", "domain", "split", "history", "supplied_context",
        "legal_questions", "optimal_questions", "trajectory_scores",
    }
    counts: dict[str, dict[str, list[int]]] = {}
    state_counts: dict[str, int] = {}
    for row in examples:
        if set(row) != expected_schema or row["domain"] not in protocol.DOMAINS:
            raise ValueError("Invalid training-example schema.")
        history = tuple(tuple(item) for item in row["history"])
        domain, supplied = row["domain"], row["supplied_context"]
        key = (
            memoryless_state_key(history, supplied, domain)
            if memoryless else full_state_key(history, supplied, domain)
        )
        legal = tuple(row["legal_questions"])
        optimal = set(row["optimal_questions"])
        if not optimal or not optimal.issubset(legal):
            raise ValueError("Training label must contain legal optimal questions.")
        state_counts[key] = state_counts.get(key, 0) + 1
        table = counts.setdefault(key, {})
        for question in legal:
            pair = table.setdefault(question, [0, 0])
            pair[1] += 1
            pair[0] += int(question in optimal)
    tables = {
        key: {
            question: {
                "positive": positive, "opportunities": opportunities,
                "score": (positive + 1) / (opportunities + 2),
            }
            for question, (positive, opportunities) in sorted(rows.items())
        }
        for key, rows in sorted(counts.items())
    }
    body = {
        "version": MODEL_VERSION, "memoryless": memoryless, "tables": tables,
        "state_counts": dict(sorted(state_counts.items())), "provenance": provenance,
    }
    return PreferenceModel(**body, model_sha256=digest(body))


def _validate_model(document: dict) -> PreferenceModel:
    if set(document) != {
        "version", "memoryless", "tables", "state_counts", "provenance", "model_sha256",
    }:
        raise ValueError("Invalid model schema.")
    body = {key: value for key, value in document.items() if key != "model_sha256"}
    if document["version"] != MODEL_VERSION:
        raise ValueError("Incompatible model version.")
    if type(document["memoryless"]) is not bool or digest(body) != document["model_sha256"]:
        raise ValueError("Invalid model content digest.")
    provenance = document["provenance"]
    required = {
        "fit_split", "protocol_sha256", "training_examples_sha256",
        "training_example_count", "training_episode_ids", "training_physical_world_ids", "label_rule",
    }
    if (
        set(provenance) != required or provenance["fit_split"] != "train"
        or provenance["protocol_sha256"] != protocol.PROTOCOL_SHA256
        or type(provenance["training_example_count"]) is not int
        or provenance["training_example_count"] <= 0
        or provenance["training_episode_ids"] != sorted(set(provenance["training_episode_ids"]))
        or provenance["training_physical_world_ids"] != sorted(set(provenance["training_physical_world_ids"]))
    ):
        raise ValueError("Invalid model provenance.")
    if (
        type(document["tables"]) is not dict or not document["tables"]
        or type(document["state_counts"]) is not dict
        or set(document["tables"]) != set(document["state_counts"])
    ):
        raise ValueError("Invalid model state tables.")
    for key, table in document["tables"].items():
        if type(key) is not str or type(document["state_counts"][key]) is not int:
            raise ValueError("Invalid model state count.")
        if type(table) is not dict or not table or any(q not in QUESTIONS for q in table):
            raise ValueError("Invalid model question table.")
        for values in table.values():
            if set(values) != {"positive", "opportunities", "score"}:
                raise ValueError("Invalid model score schema.")
            positive, opportunities, score = values["positive"], values["opportunities"], values["score"]
            if (
                type(positive) is not int or type(opportunities) is not int
                or not 0 <= positive <= opportunities
                or type(score) not in (int, float) or isinstance(score, bool)
                or not math.isfinite(score) or score != (positive + 1) / (opportunities + 2)
            ):
                raise ValueError("Invalid model score.")
    return PreferenceModel(**document)


def save_model(model: PreferenceModel, path: Path) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(model.document(), sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_model(path: Path) -> PreferenceModel:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing model: {path}.")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError("Corrupt model JSON.") from error
    return _validate_model(document)
