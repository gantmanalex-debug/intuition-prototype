"""Preregistered context-first, two-domain finite inquiry benchmark.

This module declares the ontology, questions, split generator, and target
rules.  Importing it never fits a model or executes a policy.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import random

from intuition_prototype.benchmark_protocol import digest


VERSION = "abstraction-inquiry-v3-required-means"
GENERATOR_SEED = 2026090903
DOMAINS = ("dog", "linked_list")
MAX_PHYSICAL_QUESTIONS = 4
MAX_TOTAL_QUESTIONS = 5
MAX_QUESTIONS = MAX_TOTAL_QUESTIONS
PHYSICAL_ANSWERS = ("no", "yes", "unknown")

CONTEXT_QUESTION_BY_DOMAIN = {
    "dog": "dog_clarify_goal",
    "linked_list": "linked_clarify_goal",
}
CONTEXT_QUESTION_TEXT = {
    "dog": (
        "What are you trying to determine, and what prompted the question? "
        "Choose: explain the observed pattern; assess outside-directed interest; "
        "characterize an observed change; or unknown/not supplied."
    ),
    "linked_list": (
        "What underlying operation or decision must the requested linked list serve? "
        "Choose: recommend a representation from operation constraints; assess whether "
        "node-linked representation fits; assess indexed-access needs; characterize the "
        "operation profile; implement or study a linked list because that structure "
        "is explicitly required; or unknown/not supplied."
    ),
}

# Concern is represented separately from goal.  In particular, "worried" is
# supplied concern, never evidence of physical discomfort or a diagnosis.
CONTEXTS = {
    "dog_curious_explanation": {
        "domain": "dog", "goal": "explain_pattern", "concern": "none",
        "description": "A curious observer wants the finest supported pattern description.",
    },
    "dog_worried_explanation": {
        "domain": "dog", "goal": "explain_pattern", "concern": "worried",
        "description": "The asker is worried and wants an explanation; worry is not physical evidence.",
    },
    "dog_practical_outside": {
        "domain": "dog", "goal": "assess_outside", "concern": "none",
        "description": "The asker practically wants an outside-directed assessment.",
    },
    "dog_observed_change": {
        "domain": "dog", "goal": "characterize_change", "concern": "observed_change",
        "description": "A familiar observer wants the broad observed change characterized.",
    },
    "linked_recommend_from_requirements": {
        "domain": "linked_list", "goal": "recommend_representation", "concern": "requested_linked_list",
        "description": "The requested means is known, but the desired result is a requirements-based choice.",
    },
    "linked_assess_fit": {
        "domain": "linked_list", "goal": "assess_linked_fit", "concern": "requested_linked_list",
        "description": "The asker wants to know whether the operation profile supports a node-linked family.",
    },
    "linked_assess_indexing": {
        "domain": "linked_list", "goal": "assess_indexing", "concern": "none",
        "description": "The asker wants to establish whether frequent indexed access is required.",
    },
    "linked_operation_profile": {
        "domain": "linked_list", "goal": "characterize_operations", "concern": "none",
        "description": "The asker wants the broad operation-constraint family characterized.",
    },
    "linked_required_structure": {
        "domain": "linked_list", "goal": "honor_required_structure",
        "concern": "explicit_structure_requirement",
        "description": (
            "The linked-list structure is explicitly required for implementation or education, "
            "not a tentative choice to replace with another representation."
        ),
    },
}
CONTEXT_ANSWERS_BY_DOMAIN = {
    domain: tuple(key for key, value in CONTEXTS.items() if value["domain"] == domain) + ("unknown",)
    for domain in DOMAINS
}
CONTEXT_VARIANTS = {
    "dog": (
        {"variant": "explicit_curious", "answer": "dog_curious_explanation", "explicit": True},
        {"variant": "explicit_worried", "answer": "dog_worried_explanation", "explicit": True},
        {"variant": "explicit_outside", "answer": "dog_practical_outside", "explicit": True},
        {"variant": "explicit_change", "answer": "dog_observed_change", "explicit": True},
        {"variant": "clarified_curious", "answer": "dog_curious_explanation", "explicit": False},
        {"variant": "ambiguous_intent", "answer": "unknown", "explicit": False},
    ),
    "linked_list": (
        {"variant": "explicit_recommend", "answer": "linked_recommend_from_requirements", "explicit": True},
        {"variant": "explicit_fit", "answer": "linked_assess_fit", "explicit": True},
        {"variant": "explicit_indexing", "answer": "linked_assess_indexing", "explicit": True},
        {"variant": "explicit_profile", "answer": "linked_operation_profile", "explicit": True},
        {"variant": "clarified_requested_means", "answer": "linked_recommend_from_requirements", "explicit": False},
        {"variant": "ambiguous_requirement", "answer": "unknown", "explicit": False},
        {"variant": "explicit_required_structure", "answer": "linked_required_structure", "explicit": True},
        {"variant": "clarified_required_structure", "answer": "linked_required_structure", "explicit": False},
    ),
}

GROUPS_BY_DOMAIN = {
    "dog": {
        "outside_directed": (
            "outside_orient_then_wait", "outside_pace_to_entry",
            "outside_cue_focus", "outside_quiet_watch",
        ),
        "recent_transition": (
            "return_then_rest", "return_then_scan",
            "entry_pause_then_settle", "entry_pause_then_move_on",
        ),
        "person_linked": (
            "departure_watch", "arrival_wait", "person_cue_tracking", "quiet_person_wait",
        ),
        "nonspecific_change": (
            "changed_low_activity", "changed_repositioning",
            "changed_rest_location", "changed_entry_pause",
        ),
    },
    "linked_list": {
        "sequential_collection": (
            "append_and_iterate", "front_back_queue", "insertion_order_log", "streaming_sequence",
        ),
        "indexed_collection": (
            "frequent_index_lookup", "dense_position_updates", "compact_random_access", "snapshot_indexing",
        ),
        "stable_identity": (
            "stable_external_handles", "persistent_cursors", "intrusive_membership", "node_reference_updates",
        ),
        "splice_mutation": (
            "known_node_removal", "middle_splice", "constant_time_concatenation", "split_merge_sequences",
        ),
    },
}
LEAVES_BY_DOMAIN = {
    domain: tuple(leaf for leaves in groups.values() for leaf in leaves)
    for domain, groups in GROUPS_BY_DOMAIN.items()
}
GROUPS = tuple(group for domain in DOMAINS for group in GROUPS_BY_DOMAIN[domain])
LEAVES = tuple(leaf for domain in DOMAINS for leaf in LEAVES_BY_DOMAIN[domain])

PHYSICAL_QUESTIONS_BY_DOMAIN = {
    "dog": {
        "dog_active_entry_pattern": "Does the observed sequence include active movement at the entry rather than only quiet waiting?",
        "dog_outdoor_cue_response": "Within an active-entry sequence, does a routine outdoor cue increase door-directed behavior?",
        "dog_person_timing_link": "Within a quiet-entry sequence, did the pattern begin around a familiar person's departure or arrival?",
        "dog_outside_signature": "Across the observation, are both orientation and response consistently directed to the outdoor side?",
        "dog_ambient_entry_sound": "Was an unrelated ambient entry sound present during the observation?",
        "dog_upper_sequence_repeat": "Does the first half of the observed sequence repeat?",
        "dog_lower_sequence_repeat": "Does the second half of the observed sequence repeat?",
        "dog_family_specific_pause": "Within the supported family, is its distinctive first-variant pause sequence present?",
    },
    "linked_list": {
        "linked_identity_or_splice": "Must the collection preserve stable element handles or support cheap known-position splicing?",
        "linked_fast_index": "Is frequent direct access by numeric position an underlying requirement?",
        "linked_stable_handles": "Within handle-or-splice needs, must external handles remain stable across unrelated updates?",
        "linked_known_position_updates": "Are updates normally given an existing element handle rather than a numeric position?",
        "linked_contiguous_locality": "Is compact contiguous storage locality an explicit priority?",
        "linked_cross_sequence_move": "Must ranges move between sequences without copying their elements?",
        "linked_upper_operation_repeat": "Does the first half of the operation profile recur frequently?",
        "linked_lower_operation_repeat": "Does the second half of the operation profile recur frequently?",
    },
}
QUESTIONS = {
    **{CONTEXT_QUESTION_BY_DOMAIN[d]: CONTEXT_QUESTION_TEXT[d] for d in DOMAINS},
    **{q: text for d in DOMAINS for q, text in PHYSICAL_QUESTIONS_BY_DOMAIN[d].items()},
}
PHYSICAL_QUESTION_ORDER_BY_DOMAIN = {
    domain: tuple(questions) for domain, questions in PHYSICAL_QUESTIONS_BY_DOMAIN.items()
}
QUESTION_ORDER_BY_DOMAIN = {
    domain: (CONTEXT_QUESTION_BY_DOMAIN[domain],) + PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    for domain in DOMAINS
}
CONTEXT_QUESTIONS = tuple(CONTEXT_QUESTION_BY_DOMAIN.values())
PHYSICAL_QUESTION_ORDER = tuple(q for d in DOMAINS for q in PHYSICAL_QUESTION_ORDER_BY_DOMAIN[d])
QUESTION_ORDER = tuple(q for d in DOMAINS for q in QUESTION_ORDER_BY_DOMAIN[d])
POLICIES = ("fixed_order", "memoryless_learned", "learned_history", "information_gain", "reference")

PHYSICAL_UNITS_BY_DOMAIN = {split: {domain: 19 for domain in DOMAINS} for split in ("train", "validation", "test")}
PHYSICAL_UNITS = {split: sum(values.values()) for split, values in PHYSICAL_UNITS_BY_DOMAIN.items()}
COUNTS = {
    split: sum(PHYSICAL_UNITS_BY_DOMAIN[split][domain] * len(CONTEXT_VARIANTS[domain]) for domain in DOMAINS)
    for split in PHYSICAL_UNITS_BY_DOMAIN
}


def _check_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown inquiry domain: {domain!r}.")


def leaf_group(domain: str, leaf: str) -> str:
    _check_domain(domain)
    for group, leaves in GROUPS_BY_DOMAIN[domain].items():
        if leaf in leaves:
            return group
    raise ValueError(f"Unknown {domain} world leaf: {leaf!r}.")


def _bits(domain: str, leaf: str) -> tuple[int, int]:
    group = leaf_group(domain, leaf)
    index = GROUPS_BY_DOMAIN[domain][group].index(leaf)
    return index // 2, index % 2


def allowed_answers(domain: str, leaf: str, question: str) -> tuple[str, ...]:
    """Binary values allowed by at least one declared nuisance-world."""
    _check_domain(domain)
    if question not in PHYSICAL_QUESTIONS_BY_DOMAIN[domain]:
        raise ValueError(f"Not a {domain} physical question: {question!r}.")
    group = leaf_group(domain, leaf)
    upper, lower = _bits(domain, leaf)
    if domain == "dog":
        if question == "dog_active_entry_pattern":
            return ("yes",) if group in ("outside_directed", "recent_transition") else ("no",)
        if question == "dog_outdoor_cue_response":
            if group == "outside_directed":
                return ("yes",)
            if group == "recent_transition":
                return ("no",)
            return ("no", "yes")
        if question == "dog_person_timing_link":
            if group == "person_linked":
                return ("yes",)
            if group == "nonspecific_change":
                return ("no",)
            return ("no", "yes")
        if question == "dog_outside_signature":
            return ("yes",) if group == "outside_directed" else ("no",)
        if question == "dog_ambient_entry_sound":
            return ("no", "yes")
        if question == "dog_upper_sequence_repeat":
            return ("yes",) if upper else ("no",)
        if question == "dog_lower_sequence_repeat":
            return ("yes",) if lower else ("no",)
        if question == "dog_family_specific_pause":
            return ("yes",) if GROUPS_BY_DOMAIN[domain][group].index(leaf) == 0 else ("no",)
    else:
        if question == "linked_identity_or_splice":
            return ("yes",) if group in ("stable_identity", "splice_mutation") else ("no",)
        if question == "linked_fast_index":
            if group == "indexed_collection":
                return ("yes",)
            if group == "sequential_collection":
                return ("no",)
            return ("no", "yes")
        if question == "linked_stable_handles":
            if group == "stable_identity":
                return ("yes",)
            if group == "splice_mutation":
                return ("no",)
            return ("no", "yes")
        if question == "linked_known_position_updates":
            return ("no", "yes")
        if question == "linked_contiguous_locality":
            return ("yes",) if group == "indexed_collection" else ("no",)
        if question == "linked_cross_sequence_move":
            return ("yes",) if group == "splice_mutation" else ("no",)
        if question == "linked_upper_operation_repeat":
            return ("yes",) if upper else ("no",)
        if question == "linked_lower_operation_repeat":
            return ("yes",) if lower else ("no",)
    raise ValueError(f"Unknown {domain} physical question: {question!r}.")


def canonical_pattern(domain: str, leaf: str, nuisance_bits: tuple[int, ...]) -> tuple[str, ...]:
    questions = PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    wildcards = [q for q in questions if len(allowed_answers(domain, leaf, q)) == 2]
    if len(nuisance_bits) != len(wildcards):
        raise ValueError("Wrong number of nuisance bits.")
    nuisance = dict(zip(wildcards, nuisance_bits, strict=True))
    return tuple(
        allowed_answers(domain, leaf, question)[nuisance[question]]
        if len(allowed_answers(domain, leaf, question)) == 2
        else allowed_answers(domain, leaf, question)[0]
        for question in questions
    )


PROTOCOL = {
    "version": VERSION,
    "task": (
        "Clarify the supplied goal and concern before narrowing fictional dog observations or "
        "software operation requirements. Questions are observations, not repairs or external-tool actions."
    ),
    "domains": {
        "dog": {
            "opening": (
                'The utterance is: "Why is the dog sitting near the door?" The scene does not reveal '
                "whether the asker is curious, worried, practically assessing outside interest, or describing change."
            ),
            "safety": "Toy observed-pattern descriptions only; no health diagnosis, treatment, or discomfort inference.",
        },
        "linked_list": {
            "opening": (
                'The utterance is: "Implement a linked list." A requested means does not establish the '
                "underlying ordered-collection, stable-handle, indexed-access, or splicing requirements."
            ),
            "safety": "Clarify operation constraints before recommending a representation; do not assume linked list is the answer.",
        },
    },
    "contexts": CONTEXTS,
    "context_variants": CONTEXT_VARIANTS,
    "context_questions": CONTEXT_QUESTION_TEXT,
    "context_rules": {
        "source": "Only explicit supplied context or the controlled clarification answer is used.",
        "forbidden_inference": "Never infer identity, demographics, emotions, goals, concern, or physical state from wording.",
        "separation": (
            "Goal and concern affect learned relevance, question ranking, stopping, scope, and answer detail only. "
            "They never alter compatible physical/requirement groups or leaves."
        ),
        "unknown": "Unknown goal receives one lawful clarifier; an unknown answer then abstains without an assumed goal.",
        "explicit": "Explicitly known context skips the redundant clarifier.",
    },
    "groups": {d: {g: list(v) for g, v in GROUPS_BY_DOMAIN[d].items()} for d in DOMAINS},
    "questions": {d: {q: QUESTIONS[q] for q in QUESTION_ORDER_BY_DOMAIN[d]} for d in DOMAINS},
    "world_answer_semantics": {
        "yes_no": "Fixed observed attributes or operation constraints; questions never change answers.",
        "unknown": "Not observed/supplied; imposes no compatibility constraint.",
        "out_of_model": "Any other token causes visible abstention.",
        "compatibility": "A leaf survives iff every known answer is in its declared allowed-answer set.",
    },
    "inference": {
        "prior": "No probability prior; explicit compatible sets only.",
        "entropy": "log2(candidate leaves) under a declared uniform-count convention, not calibrated probability.",
        "goal_resolution": {
            "dog/explain_pattern": "one fine leaf",
            "dog/assess_outside": "outside-directed membership constant across candidates",
            "dog/characterize_change": "one broad group",
            "linked_list/recommend_representation": "one fine operation profile",
            "linked_list/assess_linked_fit": "node-linked suitability projection constant across candidates",
            "linked_list/assess_indexing": "indexed-access projection constant across candidates",
            "linked_list/characterize_operations": "one broad operation group",
            "linked_list/honor_required_structure": (
                "explicit or clarified linked-list implementation/education requirement; "
                "acknowledge scope without claiming code generation or operation-profile evidence"
            ),
        },
        "limitation": "Contradictions and unsupported tokens are detected; unmodelled worlds that mimic a declared leaf are not.",
    },
    "budget": {
        "physical_questions": MAX_PHYSICAL_QUESTIONS,
        "total_when_context_unknown": MAX_TOTAL_QUESTIONS,
        "explicit_context": "Clarifier costs zero when context is supplied; four physical questions remain.",
        "no_minimum_gate": True,
        "repeats": "illegal",
    },
    "target": (
        "Development labels prioritize eventual scoped resolution, early relevant projection reduction, "
        "broad/fine compatible-count reduction, and cost. They do not reward instant hidden-label payoff."
    ),
    "reference": "Evaluator-only exhaustive future-answer search; future answers never enter the agent API.",
    "generator": {
        "seed": GENERATOR_SEED,
        "physical_units_by_domain": PHYSICAL_UNITS_BY_DOMAIN,
        "row_counts": COUNTS,
        "crossing": "Every physical world is crossed with every domain context variant and kept in one split.",
        "partition": "Distinct deterministic nuisance patterns per leaf/split; zero exact domain-pattern overlap.",
        "statistical_unit": "physical_world_id with all context variants",
        "structural_overlap": "Ontology and mechanisms overlap; held-out nuisance patterns are conditional generalization.",
    },
    "fit": {
        "split": "train physical units and crossed context variants only",
        "learner": (
            "Deterministic add-one-smoothed evaluator-optimal question counts. History keys goal, concern, "
            "asked set, and full compatible state; the ablation keeps goal/concern and latest answer only."
        ),
        "validation": "Report only; no test tuning or refit.",
        "unsupported": "Unseen learned states explicitly abstain; there is no fallback.",
    },
    "policies": {
        "fixed_order": {d: list(QUESTION_ORDER_BY_DOMAIN[d]) for d in DOMAINS},
        "memoryless_learned": "learned latest-answer ablation",
        "learned_history": "train-only context-conditioned compatible-history ranking",
        "information_gain": "nonlearned public scoped uniform compatible-count information",
        "reference": "evaluator-only future-answer comparator",
    },
    "endpoints": [
        "context_clarification_cost", "goal_resolution_and_correctness",
        "groups_and_leaves_remaining_by_step", "uniform_count_entropy_reduction",
        "full_leaf_resolution", "abstention_and_unsupported_answer",
        "question_cost_legality_repeats", "optimal_tie_agreement",
    ],
    "test_rule": (
        "Freeze only after parent review. Official test requires explicit one-use authorization. "
        "No publication claim or mandatory winner."
    ),
}
# Keep the declaration byte-stable across JSON archival/readback (notably,
# tuples used by Python constants become arrays in the sealed document).
PROTOCOL = json.loads(json.dumps(PROTOCOL, sort_keys=True))
PROTOCOL_SHA256 = digest(PROTOCOL)


def _special_worlds(domain: str, split: str) -> list[dict]:
    split_index = tuple(PHYSICAL_UNITS_BY_DOMAIN).index(split)
    leaf = LEAVES_BY_DOMAIN[domain][0]
    questions = PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
    width = sum(len(allowed_answers(domain, leaf, q)) == 2 for q in questions)
    variants = list(product((0, 1), repeat=width))
    base = list(canonical_pattern(domain, leaf, variants[split_index % len(variants)]))
    unknown = list(base)
    unknown[questions.index(
        "dog_upper_sequence_repeat" if domain == "dog" else "linked_upper_operation_repeat"
    )] = "unknown"
    inconsistent = list(base)
    if domain == "dog":
        inconsistent[questions.index("dog_active_entry_pattern")] = "yes"
        inconsistent[questions.index("dog_outdoor_cue_response")] = "yes"
        inconsistent[questions.index("dog_outside_signature")] = "no"
    else:
        inconsistent[questions.index("linked_identity_or_splice")] = "yes"
        inconsistent[questions.index("linked_stable_handles")] = "yes"
        inconsistent[questions.index("linked_contiguous_locality")] = "yes"
        inconsistent[questions.index("linked_fast_index")] = ("no", "no", "yes")[split_index]
        inconsistent[questions.index("linked_known_position_updates")] = ("no", "yes", "no")[split_index]
    outside = list(base)
    outside[0] = "unlisted"
    return [
        {"case_type": "physical_unknown", "domain": domain, "leaf": leaf, "answers": unknown},
        {"case_type": "inconsistent", "domain": domain, "leaf": leaf, "answers": inconsistent},
        {"case_type": "out_of_model", "domain": domain, "leaf": leaf, "answers": outside},
    ]


def _physical_units(seed: int) -> dict[str, list[dict]]:
    result = {split: [] for split in PHYSICAL_UNITS_BY_DOMAIN}
    for domain_index, domain in enumerate(DOMAINS):
        for leaf_index, leaf in enumerate(LEAVES_BY_DOMAIN[domain]):
            questions = PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain]
            width = sum(len(allowed_answers(domain, leaf, q)) == 2 for q in questions)
            variants = list(product((0, 1), repeat=width))
            random.Random(seed + domain_index * 1_000_003 + leaf_index * 104729).shuffle(variants)
            for split_index, split in enumerate(PHYSICAL_UNITS_BY_DOMAIN):
                result[split].append({
                    "case_type": "modelled", "domain": domain, "leaf": leaf,
                    "answers": list(canonical_pattern(domain, leaf, variants[split_index])),
                })
        for split in PHYSICAL_UNITS_BY_DOMAIN:
            result[split].extend(_special_worlds(domain, split))
    for split, rows in result.items():
        random.Random(seed + tuple(PHYSICAL_UNITS_BY_DOMAIN).index(split) * 65537).shuffle(rows)
        for row in rows:
            row["group"] = leaf_group(row["domain"], row["leaf"])
            row["physical_world_id"] = digest({
                "version": VERSION, "split": split, "domain": row["domain"],
                "case_type": row["case_type"], "leaf": row["leaf"], "answers": row["answers"],
            })[:20]
    return result


def generate_episodes(seed: int = GENERATOR_SEED) -> dict[str, list[dict]]:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Generator seed must be an integer in [0, 2**63).")
    result = {split: [] for split in PHYSICAL_UNITS_BY_DOMAIN}
    for split, units in _physical_units(seed).items():
        for unit in units:
            for context in CONTEXT_VARIANTS[unit["domain"]]:
                answer = context["answer"]
                data = CONTEXTS.get(answer)
                row = {
                    **unit, "split": split, "context_variant": context["variant"],
                    "context_explicit": context["explicit"], "context_answer": answer,
                    "goal": data["goal"] if data else None,
                    "concern": data["concern"] if data else None,
                }
                row["episode_id"] = digest({
                    "physical_world_id": unit["physical_world_id"],
                    "context_variant": context["variant"],
                })[:20]
                result[split].append(row)
    validate_partitions(result)
    return result


def validate_partitions(partitions: dict[str, list[dict]]) -> None:
    if set(partitions) != set(COUNTS) or {s: len(v) for s, v in partitions.items()} != COUNTS:
        raise ValueError("Partition names or counts changed.")
    expected_keys = {
        "episode_id", "physical_world_id", "split", "case_type", "domain", "group", "leaf",
        "answers", "context_variant", "context_explicit", "context_answer", "goal", "concern",
    }
    episode_ids: set[str] = set()
    world_split: dict[str, str] = {}
    patterns: dict[tuple[str, tuple[str, ...]], str] = {}
    for split, rows in partitions.items():
        by_world: dict[str, list[dict]] = {}
        for row in rows:
            domain = row.get("domain")
            if (
                set(row) != expected_keys or row["episode_id"] in episode_ids or domain not in DOMAINS
                or row["split"] != split or row["leaf"] not in LEAVES_BY_DOMAIN[domain]
                or row["group"] != leaf_group(domain, row["leaf"])
                or len(row["answers"]) != len(PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain])
            ):
                raise ValueError("Invalid generated episode schema or identity.")
            episode_ids.add(row["episode_id"])
            by_world.setdefault(row["physical_world_id"], []).append(row)
            if world_split.setdefault(row["physical_world_id"], split) != split:
                raise ValueError("A physical statistical unit crossed partitions.")
        if len(by_world) != PHYSICAL_UNITS[split]:
            raise ValueError("Physical-unit count changed.")
        for variants in by_world.values():
            domain = variants[0]["domain"]
            if {row["context_variant"] for row in variants} != {
                item["variant"] for item in CONTEXT_VARIANTS[domain]
            }:
                raise ValueError("A physical world is missing a crossed context.")
            if len({tuple(row["answers"]) for row in variants}) != 1 or len({row["leaf"] for row in variants}) != 1:
                raise ValueError("Context crossing changed the physical world.")
            key = (domain, tuple(variants[0]["answers"]))
            if patterns.setdefault(key, split) != split:
                raise ValueError("Exact domain physical-answer pattern overlaps partitions.")
        for domain in DOMAINS:
            exceptional = {
                kind: len({row["physical_world_id"] for row in rows
                           if row["domain"] == domain and row["case_type"] == kind})
                for kind in ("physical_unknown", "inconsistent", "out_of_model")
            }
            if exceptional != {"physical_unknown": 1, "inconsistent": 1, "out_of_model": 1}:
                raise ValueError("Exceptional physical-unit coverage changed.")


def source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "src/intuition_prototype/abstraction_protocol.py",
        "src/intuition_prototype/abstraction_inquiry.py",
        "src/intuition_prototype/abstraction_benchmark.py",
        "src/intuition_prototype/benchmark_protocol.py",
        "src/intuition_prototype/stage4_protocol.py",
        "src/intuition_prototype/records.py",
        "src/intuition_prototype/simulator.py",
        "pyproject.toml",
    )
    return {name: root / name for name in names}
