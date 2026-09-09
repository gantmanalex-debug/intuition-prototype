"""Development-only tests; no official abstraction test is executed here."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil

import pytest

from intuition_prototype import abstraction_benchmark as benchmark
from intuition_prototype import abstraction_inquiry as task
from intuition_prototype import abstraction_protocol as protocol


def _episode(row):
    return benchmark._episode(row)


@pytest.fixture(scope="module")
def fitted():
    partitions = protocol.generate_episodes()
    examples = benchmark.training_examples(partitions["train"])
    provenance = benchmark._provenance(examples)
    model = task.fit_preference(examples, provenance)
    memoryless = task.fit_preference(examples, provenance, memoryless=True)
    return partitions, examples, model, memoryless


def test_preregistered_two_domain_splits_are_grouped_and_disjoint():
    partitions = protocol.generate_episodes()
    assert partitions == protocol.generate_episodes()
    assert partitions != protocol.generate_episodes(protocol.GENERATOR_SEED + 1)
    assert {split: len(rows) for split, rows in partitions.items()} == protocol.COUNTS
    patterns = {
        split: {(row["domain"], tuple(row["answers"])) for row in rows}
        for split, rows in partitions.items()
    }
    assert patterns["train"].isdisjoint(patterns["validation"] | patterns["test"])
    assert patterns["validation"].isdisjoint(patterns["test"])
    for split, rows in partitions.items():
        worlds = defaultdict(list)
        for row in rows:
            worlds[row["physical_world_id"]].append(row)
        assert len(worlds) == protocol.PHYSICAL_UNITS[split]
        for variants in worlds.values():
            domain = variants[0]["domain"]
            assert {row["context_variant"] for row in variants} == {
                item["variant"] for item in protocol.CONTEXT_VARIANTS[domain]
            }
            assert len({tuple(row["answers"]) for row in variants}) == 1
        for domain in protocol.DOMAINS:
            assert {
                kind: len({r["physical_world_id"] for r in rows
                           if r["domain"] == domain and r["case_type"] == kind})
                for kind in ("physical_unknown", "inconsistent", "out_of_model")
            } == {"physical_unknown": 1, "inconsistent": 1, "out_of_model": 1}


@pytest.mark.parametrize("domain", protocol.DOMAINS)
def test_context_only_never_changes_compatible_worlds(domain):
    before = task.candidate_state((), domain)
    context = next(key for key, data in protocol.CONTEXTS.items() if data["domain"] == domain)
    question = protocol.CONTEXT_QUESTION_BY_DOMAIN[domain]
    after = task.candidate_state(((question, context),), domain)
    assert after["candidate_groups"] == before["candidate_groups"]
    assert after["candidate_leaves"] == before["candidate_leaves"]
    trace = task.evidence_trace(((question, context),), domain=domain)[0]
    assert trace["kind"] == "context_clarification"
    assert trace["group_count_reduction"] == trace["leaf_count_reduction"] == 0
    assert trace["interpretation"] == "context only; compatible worlds unchanged"


def test_unknown_context_clarifies_and_explicit_context_skips_redundancy(fitted):
    partitions = fitted[0]
    for domain in protocol.DOMAINS:
        ambiguous = next(
            row for row in partitions["train"]
            if row["domain"] == domain and not row["context_explicit"]
            and row["context_answer"] == "unknown"
        )
        api = task.AbstractionQuestionAPI(_episode(ambiguous))
        assert api.legal_questions() == (protocol.CONTEXT_QUESTION_BY_DOMAIN[domain],)
        api.ask(api.legal_questions()[0])
        assert api.observe()["inquiry_state"]["context"]["status"] == "ambiguous"
        assert api.legal_questions() == ()

        explicit = next(
            row for row in partitions["train"]
            if row["domain"] == domain and row["context_explicit"]
        )
        api = task.AbstractionQuestionAPI(_episode(explicit))
        assert protocol.CONTEXT_QUESTION_BY_DOMAIN[domain] not in api.legal_questions()
        assert api.observe()["remaining_physical_budget"] == protocol.MAX_PHYSICAL_QUESTIONS


def test_dog_worry_is_separate_context_not_discomfort_evidence():
    worried = task.context_state((), "dog_worried_explanation", "dog")
    curious = task.context_state((), "dog_curious_explanation", "dog")
    assert worried["goal"] == curious["goal"] == "explain_pattern"
    assert worried["concern"] == "worried" and curious["concern"] == "none"
    assert task.candidate_state((), "dog") == task.candidate_state(
        ((protocol.CONTEXT_QUESTION_BY_DOMAIN["dog"], "dog_worried_explanation"),), "dog"
    )
    text = json.dumps(protocol.PROTOCOL["contexts"]["dog_worried_explanation"]).lower()
    assert "discomfort" not in text and "diagnos" not in text


def test_requested_linked_list_is_a_means_and_clarifier_asks_requirements():
    opening = task.OPENINGS["linked_list"].lower()
    clarifier = protocol.CONTEXT_QUESTION_TEXT["linked_list"].lower()
    assert "requested means" in opening
    assert "does not establish" in opening
    assert "underlying operation" in clarifier
    assert "assume linked list is the answer" in protocol.PROTOCOL["domains"]["linked_list"]["safety"]


@pytest.mark.parametrize("explicit", [True, False])
def test_required_linked_list_terminates_without_reinterpreting_world(explicit):
    row = next(
        row for row in protocol.generate_episodes()["train"]
        if row["domain"] == "linked_list"
        and row["context_answer"] == "linked_required_structure"
        and row["context_explicit"] == explicit
        and row["case_type"] == "modelled"
    )
    episode = _episode(row)
    api = task.AbstractionQuestionAPI(episode)
    initial_world = api.observe()["inquiry_state"]["world"]
    if not explicit:
        assert api.legal_questions() == ("linked_clarify_goal",)
        api.ask("linked_clarify_goal")
    state = api.observe()["inquiry_state"]
    assert state["world"] == initial_world
    assert state["world"]["leaf_count"] == 16
    assert not state["world"]["supported"]
    assert state["goal_resolution"]["supported"]
    assert state["goal_resolution"]["value"] == "linked_list_required"
    assert state["goal_resolution"]["scope"] == "explicit_structure_requirement"
    assert "has not generated" in state["goal_resolution"]["answer"]
    assert api.legal_questions() == ()
    reference = benchmark.reference(episode)
    assert reference["minimum_questions"] == (0 if explicit else 1)
    assert reference["shortest_paths"] == ([[]] if explicit else [["linked_clarify_goal"]])
    result = benchmark.run_policy(episode, "fixed_order", None, None)
    assert result["correct"] is True and result["goal_resolved"]
    assert not result["full_leaf_resolved"]
    assert result["question_cost"] == (0 if explicit else 1)


def test_required_means_is_positive_option_not_automatic_inference():
    assert "explicitly required" in protocol.CONTEXT_QUESTION_TEXT["linked_list"]
    assert "linked_required_structure" in protocol.CONTEXT_ANSWERS_BY_DOMAIN["linked_list"]
    assert not task.goal_resolution((), None, "linked_list")["supported"]
    assert not task.goal_resolution(
        (), "linked_recommend_from_requirements", "linked_list",
    )["supported"]
    assert task.legal_questions((), None, "linked_list") == ("linked_clarify_goal",)


@pytest.mark.parametrize("clarified", [False, True])
def test_required_linked_list_cli_is_honest_scoped_acceptance(clarified, capsys):
    arguments = ["required-linked-list"] + (["--clarified"] if clarified else [])
    benchmark.main(arguments)
    result = json.loads(capsys.readouterr().out)
    assert result["code_generated"] is False
    assert result["question_cost"] == int(clarified)
    assert result["final"]["world"] == result["initial"]["world"]
    assert result["final"]["goal_resolution"]["supported"]
    assert result["legal_next_questions"] == []
    assert "not a learned benchmark result" in result["classification"]


@pytest.mark.parametrize(
    "domain,history,groups,leaves",
    [
        ("dog", (("dog_active_entry_pattern", "yes"),),
         ("outside_directed", "recent_transition"), 8),
        ("linked_list", (("linked_identity_or_splice", "yes"),),
         ("stable_identity", "splice_mutation"), 8),
    ],
)
def test_broad_pruning_is_exact_with_declared_uniform_counts(domain, history, groups, leaves):
    state = task.candidate_state(history, domain)
    assert state["candidate_groups"] == groups and state["leaf_count"] == leaves
    trace = task.evidence_trace(history, domain=domain)[0]
    assert trace["group_count_reduction"] == 2
    assert trace["leaf_count_reduction"] == 8
    assert trace["uniform_count_entropy_reduction_bits"] == pytest.approx(1)
    assert "not calibrated" in state["assumptions"]


def test_crosscut_questions_can_narrow_fine_leaves_without_broad_groups():
    for domain, question, context in (
        ("dog", "dog_upper_sequence_repeat", "dog_curious_explanation"),
        ("linked_list", "linked_upper_operation_repeat", "linked_recommend_from_requirements"),
    ):
        state = task.candidate_state(((question, "yes"),), domain)
        assert state["group_count"] == 4 and state["leaf_count"] == 8
        score = task.expected_information_gain((), question, context, domain)
        assert score["leaf_entropy_reduction"] == pytest.approx(1)
        assert score["group_entropy_reduction"] == pytest.approx(0)
        assert score["kind"].startswith("nonlearned")


def test_same_physical_world_crossed_goals_stop_at_different_scopes(fitted):
    rows = fitted[0]["train"]
    for domain, projection_context, fine_context, question in (
        ("dog", "dog_practical_outside", "dog_curious_explanation", "dog_outside_signature"),
        ("linked_list", "linked_assess_fit", "linked_recommend_from_requirements", "linked_identity_or_splice"),
    ):
        world = next(
            r["physical_world_id"] for r in rows
            if r["domain"] == domain and r["case_type"] == "modelled"
            and r["context_answer"] == fine_context
        )
        variants = {r["context_answer"]: r for r in rows if r["physical_world_id"] == world}
        assert variants[projection_context]["answers"] == variants[fine_context]["answers"]
        answer = variants[fine_context]["answers"][
            protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain].index(question)
        ]
        history = ((question, answer),)
        projected = task.goal_resolution(history, projection_context, domain)
        fine = task.goal_resolution(history, fine_context, domain)
        assert projected["supported"] and projected["retained_world_uncertainty"] > 1
        assert not fine["supported"]


def test_agent_api_hides_labels_future_answers_and_enforces_repeats(fitted):
    for domain in protocol.DOMAINS:
        row = next(
            r for r in fitted[0]["train"]
            if r["domain"] == domain and r["case_type"] == "modelled" and r["context_explicit"]
        )
        api = task.AbstractionQuestionAPI(_episode(row))
        opening = api.observe()
        text = json.dumps(opening)
        assert set(opening) == {
            "domain", "opening", "supplied_context", "history", "inquiry_state",
            "legal_questions", "remaining_total_budget", "remaining_physical_budget",
        }
        assert '"leaf"' not in text and '"answers"' not in text and '"hidden' not in text
        question = api.legal_questions()[0]
        api.ask(question)
        with pytest.raises(ValueError, match="repeated|inappropriate"):
            api.ask(question)
        with pytest.raises(ValueError, match="unknown|inappropriate"):
            api.ask(protocol.CONTEXT_QUESTION_BY_DOMAIN[
                "linked_list" if domain == "dog" else "dog"
            ])


def test_actual_fitted_first_answer_branches_in_both_domains(fitted):
    model = fitted[2]
    cases = (
        (
            "dog", "dog_curious_explanation", "dog_active_entry_pattern",
            {"no": "dog_person_timing_link", "yes": "dog_outdoor_cue_response"},
        ),
        (
            "linked_list", "linked_recommend_from_requirements", "linked_identity_or_splice",
            {"no": "linked_fast_index", "yes": "linked_stable_handles"},
        ),
    )
    for domain, context, first_expected, branches in cases:
        legal = task.legal_questions((), context, domain)
        first, detail = model.choose((), legal, context, domain)
        assert first == first_expected and detail["status"] == "learned_count_ranking"
        for answer, second_expected in branches.items():
            history = ((first, answer),)
            legal = task.legal_questions(history, context, domain)
            assert model.choose(history, legal, context, domain)[0] == second_expected


def test_fitted_history_not_latest_answer_controls_both_domain_branches(fitted):
    model, memoryless = fitted[2], fitted[3]
    cases = (
        (
            "dog", "dog_curious_explanation", "dog_active_entry_pattern",
            "dog_upper_sequence_repeat", "no",
            "dog_person_timing_link", "dog_outdoor_cue_response",
        ),
        (
            "linked_list", "linked_recommend_from_requirements", "linked_identity_or_splice",
            "linked_upper_operation_repeat", "yes",
            "linked_fast_index", "linked_stable_handles",
        ),
    )
    for domain, context, earlier_question, latest_question, latest_answer, no_choice, yes_choice in cases:
        no_history = ((earlier_question, "no"), (latest_question, latest_answer))
        yes_history = ((earlier_question, "yes"), (latest_question, latest_answer))
        no_legal = task.legal_questions(no_history, context, domain)
        yes_legal = task.legal_questions(yes_history, context, domain)
        assert no_legal == yes_legal
        assert model.choose(no_history, no_legal, context, domain)[0] == no_choice
        assert model.choose(yes_history, yes_legal, context, domain)[0] == yes_choice
        assert (
            memoryless.choose(no_history, no_legal, context, domain)[0]
            == memoryless.choose(yes_history, yes_legal, context, domain)[0]
        )


def test_context_actually_changes_learned_ranking(fitted):
    model = fitted[2]
    dog_choices = {
        context: model.choose(
            (), task.legal_questions((), context, "dog"), context, "dog",
        )[0]
        for context in ("dog_curious_explanation", "dog_practical_outside")
    }
    linked_choices = {
        context: model.choose(
            (), task.legal_questions((), context, "linked_list"), context, "linked_list",
        )[0]
        for context in ("linked_assess_fit", "linked_assess_indexing")
    }
    assert len(set(dog_choices.values())) == 2
    assert len(set(linked_choices.values())) == 2


def test_unknown_inconsistent_and_out_of_model_answers_abstain_without_forced_leaf():
    cases = (
        ("dog", "dog_active_entry_pattern"),
        ("linked_list", "linked_identity_or_splice"),
    )
    for domain, question in cases:
        prefix = ((question, "yes"),)
        unknown_question = protocol.PHYSICAL_QUESTION_ORDER_BY_DOMAIN[domain][-1]
        unknown = task.candidate_state(prefix + ((unknown_question, "unknown"),), domain)
        assert unknown["candidate_leaves"] == task.candidate_state(prefix, domain)["candidate_leaves"]
        assert not unknown["supported"] and unknown["answer"] is None
        outside = task.candidate_state(((question, "violet"),), domain)
        assert outside["status"] == "out_of_model" and outside["answer"] is None

    dog_inconsistent = task.candidate_state((
        ("dog_active_entry_pattern", "yes"),
        ("dog_outdoor_cue_response", "yes"),
        ("dog_outside_signature", "no"),
    ), "dog")
    linked_inconsistent = task.candidate_state((
        ("linked_identity_or_splice", "yes"),
        ("linked_stable_handles", "yes"),
        ("linked_contiguous_locality", "yes"),
    ), "linked_list")
    for state in (dog_inconsistent, linked_inconsistent):
        assert state["status"] == "inconsistent"
        assert state["leaf_count"] == 0 and state["answer"] is None


def test_reference_certifies_shortest_scoped_depth_including_clarifier(fitted):
    rows = fitted[0]["train"]
    for domain, explicit_name, clarified_name in (
        ("dog", "explicit_curious", "clarified_curious"),
        ("linked_list", "explicit_recommend", "clarified_requested_means"),
    ):
        explicit = next(
            r for r in rows if r["domain"] == domain and r["case_type"] == "modelled"
            and r["context_variant"] == explicit_name
        )
        clarified = next(
            r for r in rows if r["physical_world_id"] == explicit["physical_world_id"]
            and r["context_variant"] == clarified_name
        )
        first = benchmark.reference(_episode(explicit))
        second = benchmark.reference(_episode(clarified))
        assert first["minimum_questions"] is not None and first["minimum_questions"] >= 3
        assert second["minimum_questions"] == first["minimum_questions"] + 1
        assert first["requires_at_least_three_distinct_questions"]
        assert first["search"]["shortcuts_rejected"] and second["search"]["shortcuts_rejected"]


def test_no_artificial_minimum_gate_allows_early_scoped_resolution():
    for domain, context, question, answer in (
        ("dog", "dog_practical_outside", "dog_outside_signature", "yes"),
        ("linked_list", "linked_assess_fit", "linked_identity_or_splice", "yes"),
    ):
        resolution = task.goal_resolution(((question, answer),), context, domain)
        assert resolution["supported"] and resolution["retained_world_uncertainty"] > 1
        assert task.legal_questions(((question, answer),), context, domain) == ()


def test_evidence_not_hidden_label_determines_projection_answer():
    yes = task.goal_resolution(
        (("dog_outside_signature", "yes"),), "dog_practical_outside", "dog",
    )
    no = task.goal_resolution(
        (("dog_outside_signature", "no"),), "dog_practical_outside", "dog",
    )
    assert yes["value"] is True and no["value"] is False
    linked_yes = task.goal_resolution(
        (("linked_identity_or_splice", "yes"),), "linked_assess_fit", "linked_list",
    )
    linked_no = task.goal_resolution(
        (("linked_identity_or_splice", "no"),), "linked_assess_fit", "linked_list",
    )
    assert linked_yes["value"] is True and linked_no["value"] is False


def test_train_only_deterministic_fit_and_strict_model_validation(fitted, tmp_path):
    partitions, examples, model, _ = fitted
    assert {row["split"] for row in examples} == {"train"}
    assert {row["domain"] for row in examples} == set(protocol.DOMAINS)
    assert set(model.provenance["training_episode_ids"]).issubset(
        {row["episode_id"] for row in partitions["train"]}
    )
    assert task.fit_preference(examples, model.provenance).model_sha256 == model.model_sha256
    with pytest.raises(ValueError, match="training-only"):
        task.fit_preference([{**examples[0], "split": "validation"}], model.provenance)
    path = tmp_path / "model.json"
    task.save_model(model, path)
    assert task.load_model(path) == model
    document = json.loads(path.read_text())
    state = next(iter(document["tables"]))
    question = next(iter(document["tables"][state]))
    document["tables"][state][question] = {"positive": 2, "opportunities": 1, "score": 1}
    body = {key: value for key, value in document.items() if key != "model_sha256"}
    document["model_sha256"] = benchmark.digest(body)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        task.load_model(path)


def test_unseen_learned_state_explicitly_abstains(fitted):
    model = fitted[2]
    altered = task.PreferenceModel(
        model.version, model.memoryless, {}, {}, model.provenance, model.model_sha256,
    )
    legal = task.legal_questions((), "dog_curious_explanation", "dog")
    question, detail = altered.choose((), legal, "dog_curious_explanation", "dog")
    assert question is None and detail["status"] == "unsupported_state_abstention"


def test_all_baselines_share_legality_budget_and_separate_hindsight(fitted):
    partitions, _, model, memoryless = fitted
    selected = [
        next(r for r in partitions["validation"]
             if r["domain"] == domain and r["case_type"] == "modelled" and r["context_explicit"])
        for domain in protocol.DOMAINS
    ]
    for row in selected:
        for policy in protocol.POLICIES:
            result = benchmark.run_policy(_episode(row), policy, model, memoryless)
            assert result["question_cost"] <= protocol.MAX_TOTAL_QUESTIONS
            assert result["budget_adherent"] and result["no_repeats"] and result["legal"]
            assert result["unsupported_answer"] == 0
            assert len(result["evidence_trace"]) == result["question_cost"]
            assert all("reference_optimal_ties" not in decision for decision in result["agent_trace"])
            assert all(annotation["source"].startswith("evaluator-only")
                       for annotation in result["reference_annotations"])
    summary = benchmark.summarize(benchmark.evaluate_rows(selected, model, memoryless))
    assert set(summary["policies"]) == set(protocol.POLICIES)


@pytest.fixture(scope="module")
def development_training(tmp_path_factory):
    directory = tmp_path_factory.mktemp("abstraction-development") / "experiment"
    prepared = benchmark.prepare(directory)
    trained = benchmark.train(directory)
    return directory, prepared, trained


def test_prepare_train_report_demo_and_both_review_guards(development_training, capsys):
    directory, prepared, trained = development_training
    assert prepared["outcomes_executed"] is False
    assert trained["validation_only"] and trained["no_test_outcomes_executed"]
    assert not (directory / "test-results.jsonl").exists()
    assert not (directory / "freeze.json").exists()
    assert benchmark.check_frozen(directory)["model_sha256"] == trained["model_sha256"]
    for domain in protocol.DOMAINS:
        shown = benchmark.demo(directory, policy="learned_history", domain=domain)
        assert shown["classification"].startswith("validation")
        assert shown["result"]["domain"] == domain
        assert shown["model_fit_provenance"]["fit_split"] == "train"
    with pytest.raises(ValueError, match="parent review"):
        benchmark.freeze(directory)
    with pytest.raises(ValueError, match="Missing sealed manifest"):
        benchmark.evaluate(directory, benchmark.OFFICIAL_AUTHORIZATION)
    assert not (directory / "official-test-started.json").exists()
    benchmark.main([
        "demo", str(directory), "--policy", "learned_history", "--domain", "linked_list",
    ])
    captured = capsys.readouterr()
    assert json.loads(captured.out)["model_sha256"] == trained["model_sha256"]
    assert json.loads(captured.out)["result"]["domain"] == "linked_list"
    assert "not calibrated" in captured.err
    assert benchmark.report(directory) == (directory / "report.md").read_text(encoding="utf-8")


def test_reviewed_freeze_still_blocks_official_test_without_one_use_authorization(
    development_training, tmp_path,
):
    directory = tmp_path / "reviewed"
    shutil.copytree(development_training[0], directory)
    frozen = benchmark.freeze(directory, benchmark.FREEZE_AUTHORIZATION)
    assert frozen["reviewed_development_freeze"] is True
    with pytest.raises(ValueError, match="official test blocked|Official test blocked"):
        benchmark.evaluate(directory)
    assert not (directory / "official-test-started.json").exists()
    assert not (directory / "test-results.jsonl").exists()


@pytest.mark.parametrize("mutation", ["extra", "missing", "modified"])
def test_strict_archived_source_inventory(development_training, tmp_path, mutation):
    directory = tmp_path / mutation
    shutil.copytree(development_training[0], directory)
    archived = directory / "sources" / "src" / "intuition_prototype" / "abstraction_inquiry.py"
    if mutation == "extra":
        (directory / "sources" / "extra.py").write_text("# extra\n", encoding="utf-8")
    elif mutation == "missing":
        archived.unlink()
    else:
        archived.write_bytes(archived.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="archive|Archived|Artifact"):
        benchmark.check_prepared(directory, execution=False)


def test_report_is_frozen_readback_not_live_render(development_training, monkeypatch):
    directory = development_training[0]
    expected = (directory / "report.md").read_text(encoding="utf-8")
    monkeypatch.setattr(
        benchmark, "render_report",
        lambda *args: pytest.fail("report unexpectedly used live renderer"),
    )
    assert benchmark.report(directory) == expected


def test_unknown_demo_episode_is_not_silently_replaced(tmp_path, monkeypatch):
    (tmp_path / "validation.json").write_text(
        json.dumps({"episodes": [{"episode_id": "known", "domain": "dog"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark, "check_frozen", lambda directory: {})
    with pytest.raises(ValueError, match="Unknown validation episode"):
        benchmark._demo_live(tmp_path, episode_id="missing", domain="dog")


def test_archived_demo_imports_archive_without_package_initializer(tmp_path):
    package = tmp_path / "sources" / "src" / "intuition_prototype"
    package.mkdir(parents=True)
    (package / "abstraction_benchmark.py").write_text(
        "def _demo_live(root, episode_id, policy, domain):\n"
        "    return {'classification': 'validation demonstration, not an official test result',\n"
        "            'archive_marker': 'isolated', 'episode_id': episode_id}\n",
        encoding="utf-8",
    )
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    shown = benchmark._archived_demo(tmp_path, "requested", "learned_history", "linked_list")
    assert shown["archive_marker"] == "isolated"
    assert shown["episode_id"] == "requested"
    assert set(shown["legacy_dependency_sha256"]) == {"records", "simulator"}
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert {
        "src/intuition_prototype/records.py", "src/intuition_prototype/simulator.py",
    } <= set(protocol.source_paths())


def test_historical_demo_uses_validated_archived_code_after_live_drift(
    development_training, monkeypatch,
):
    directory, _, trained = development_training
    before = {
        path.relative_to(directory): benchmark.file_digest(path)
        for path in directory.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(benchmark, "source_fingerprints", lambda: {"live": "drift"})
    monkeypatch.setattr(
        benchmark, "_demo_live",
        lambda *args, **kwargs: pytest.fail("historical demo used drifted live code"),
    )
    shown = benchmark.demo(directory, policy="learned_history")
    assert shown["classification"] == "validation demonstration, not an official test result"
    assert shown["model_sha256"] == trained["model_sha256"]
    after = {
        path.relative_to(directory): benchmark.file_digest(path)
        for path in directory.rglob("*") if path.is_file()
    }
    assert before == after
