"""Development-only tests for the history-aware question benchmark."""

import json
from pathlib import Path
import shutil

import pytest

from intuition_prototype import dog_history as task
from intuition_prototype import dog_history_benchmark as benchmark
from intuition_prototype import dog_history_protocol as protocol
from intuition_prototype import stage4_protocol as integrity


ACTUAL_SOURCE_PATHS = integrity.source_paths


@pytest.fixture(scope="module", autouse=True)
def stable_source_inventory(tmp_path_factory):
    """Use one exact snapshot so concurrent parent/new-version edits cannot race lifecycle tests."""
    root = tmp_path_factory.mktemp("dog-history-source-snapshot")
    paths = {}
    for name, path in ACTUAL_SOURCE_PATHS().items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        paths[name] = target
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(integrity, "source_paths", lambda: paths)
        patch.setattr(benchmark, "source_paths", lambda: paths)
        yield paths


def episode(pattern, hypothesis="wants_outside", split="train"):
    return task.HiddenEpisode("fixture", hypothesis, tuple(pattern), split)


def test_public_joint_model_and_deterministic_belief_interaction():
    assert sum(task.pattern_probability("wants_outside", p) for p in task.all_patterns()) == pytest.approx(1)
    history = (("outside_orientation", 1), ("outside_cue", 1))
    assert task.belief(history) == task.belief(history)
    assert sum(task.belief(history).values()) == pytest.approx(1)
    # The declared pair interaction makes the joint evidence non-independent.
    h = "wants_outside"
    joint = task.likelihood(h, history)
    product = task.likelihood(h, history[:1]) * task.likelihood(h, history[1:])
    assert joint != pytest.approx(product)


def test_lawful_api_reveals_only_asked_answer_and_rejects_repeat_budget():
    hidden = episode((1, 0, 1, 1, 0, 0), "waiting_for_person")
    api = task.DogQuestionAPI(hidden)
    opening = api.observe()
    assert set(opening) == {
        "opening", "history", "belief", "resolution", "legal_questions", "remaining_budget",
    }
    text = json.dumps(opening)
    assert "hidden_label" not in text and '"answers"' not in text
    assert opening["resolution"]["answer"] is None
    api.ask("outside_orientation")
    assert len(api.observe()["history"]) == 1
    with pytest.raises(ValueError, match="repeated"):
        api.ask("outside_orientation")
    for question in ("recent_return", "person_departure", "outside_cue"):
        api.ask(question)
    assert api.legal_questions() == ()
    with pytest.raises(ValueError, match="budget"):
        api.ask("comfortable_settle")


def test_same_observed_history_cannot_leak_future_answers_or_latent_label():
    first = episode((1, 0, 0, 0, 0, 0), "wants_outside")
    second = episode((1, 0, 1, 1, 1, 1), "possible_discomfort")
    left, right = task.DogQuestionAPI(first), task.DogQuestionAPI(second)
    for question in ("outside_orientation", "recent_return"):
        assert left.ask(question) == right.ask(question)
    assert left.observe() == right.observe()


def test_identical_opening_first_answer_changes_useful_next_question():
    yes = episode((1, 0, 1, 1, 1, 1), "possible_discomfort")
    no = episode((0, 0, 0, 1, 1, 0), "returned_and_resting")
    assert task.DogQuestionAPI(yes).observe()["opening"] == task.DogQuestionAPI(no).observe()["opening"]
    yes_history = (("outside_orientation", 1),)
    no_history = (("outside_orientation", 0),)
    assert benchmark.optimal_questions(yes, yes_history) == ("outside_cue",)
    assert benchmark.optimal_questions(no, no_history) == ("comfortable_settle",)


def test_same_latest_answer_different_earlier_history_requires_different_choice():
    item = episode((1, 0, 1, 1, 1, 1), "possible_discomfort")
    first = (("outside_orientation", 1), ("recent_return", 0))
    second = (("outside_cue", 1), ("recent_return", 0))
    assert first[-1] == second[-1]
    assert benchmark.optimal_questions(item, first) == ("outside_cue",)
    assert benchmark.optimal_questions(item, second) == ("outside_orientation",)


def test_exhaustive_reference_certifies_four_distinct_evidence_questions():
    item = episode((1, 1, 0, 0, 0, 1), "possible_discomfort")
    reference = benchmark.reference(item)
    assert reference["minimum_questions"] == 4
    assert reference["requires_at_least_three_distinct_questions"]
    assert reference["shortest_paths"]
    assert all(len(path) == len(set(path)) == 4 for path in reference["shortest_paths"])
    assert reference["search"]["all_distinct_prefixes_checked"] == 517
    for path in reference["shortest_paths"]:
        history = ()
        for question in path[:-1]:
            history = benchmark._extend(item, history, question)
            assert not task.resolution(history)["supported"]
        assert task.resolution(benchmark._extend(item, history, path[-1]))["supported"]


def test_answer_and_abstention_are_separate_from_guess():
    item = episode((0, 0, 0, 0, 0, 0), "waiting_for_person")
    final = task.resolution(())
    assert not final["supported"] and final["answer"] is None
    # Budget exhaustion never turns the MAP into a confident answer.
    history = tuple((q, 0) for q in task.QUESTION_ORDER[:task.MAX_QUESTIONS])
    if not task.resolution(history)["supported"]:
        assert task.resolution(history)["answer"] is None


def test_public_model_support_is_distinct_from_hidden_label_correctness():
    hidden = episode((0, 0, 0, 0, 0, 0), "wants_outside")
    api = task.DogQuestionAPI(hidden)
    for question in (
        "outside_orientation", "recent_return", "person_departure", "comfortable_settle",
    ):
        api.ask(question)
    result = api.observe()["resolution"]
    assert result["supported"] and result["answer"] == "possible_discomfort"
    assert result["answer"] != hidden.hypothesis


@pytest.fixture()
def fitted_models(tmp_path):
    rows = protocol.generate_episodes()["train"]
    examples = benchmark.training_examples(rows)
    provenance = {"fit_split": "train", "fixture": True}
    model = task.fit_preference(examples, provenance)
    memoryless = task.fit_preference(examples, provenance, memoryless=True)
    return tmp_path, model, memoryless, examples


def test_actual_transparent_fit_is_history_conditioned_and_training_only(fitted_models):
    _, model, memoryless, examples = fitted_models
    assert examples and {row["split"] for row in examples} == {"train"}
    first = (("outside_orientation", 1), ("recent_return", 0))
    second = (("outside_cue", 1), ("recent_return", 0))
    assert task.feature_names(first) != task.feature_names(second)
    assert task.feature_names(first, memoryless=True) == task.feature_names(second, memoryless=True)
    earlier_no = (("outside_orientation", 0), ("comfortable_settle", 0))
    earlier_yes = (("outside_orientation", 1), ("comfortable_settle", 0))
    legal = tuple(q for q in task.QUESTION_ORDER if q not in {"outside_orientation", "comfortable_settle"})
    no_choice, no_detail = model.choose(earlier_no, legal)
    yes_choice, yes_detail = model.choose(earlier_yes, legal)
    assert no_detail["status"] == yes_detail["status"] == "learned_preference"
    assert no_choice == "discomfort_indicator"
    assert yes_choice == "outside_cue"
    with pytest.raises(ValueError, match="training-only"):
        task.fit_preference([{**examples[0], "split": "validation"}], {})


def test_learned_opening_does_not_branch_on_its_first_answer_but_manual_starter_can(
    fitted_models,
):
    _, model, _, _ = fitted_models
    opening, _ = model.choose((), task.QUESTION_ORDER)
    assert opening == "outside_orientation"
    opening_legal = tuple(q for q in task.QUESTION_ORDER if q != opening)
    assert {
        model.choose(((opening, answer),), opening_legal)[0] for answer in (0, 1)
    } == {"comfortable_settle"}

    forced = {}
    for manual in task.QUESTION_ORDER:
        manual_legal = tuple(q for q in task.QUESTION_ORDER if q != manual)
        forced[manual] = tuple(
            model.choose(((manual, answer),), manual_legal)[0] for answer in (0, 1)
        )
    assert forced == {
        "outside_orientation": ("comfortable_settle", "comfortable_settle"),
        "recent_return": ("comfortable_settle", "comfortable_settle"),
        "person_departure": ("comfortable_settle", "comfortable_settle"),
        "outside_cue": ("comfortable_settle", "comfortable_settle"),
        "comfortable_settle": ("outside_orientation", "recent_return"),
        "discomfort_indicator": ("comfortable_settle", "comfortable_settle"),
    }


def test_train_only_demo_patterns_show_genuine_later_history_branch(fitted_models):
    _, model, memoryless, _ = fitted_models
    train = protocol.generate_episodes()["train"]
    for pattern, expected_third in (
        ((0, 1, 1, 1, 0, 0), "discomfort_indicator"),
        ((1, 0, 1, 1, 0, 1), "outside_cue"),
    ):
        row = next(row for row in train if tuple(row["answers"]) == pattern)
        shown = benchmark.run_policy(benchmark._episode(row), "learned_history", model, memoryless)
        assert [decision["chosen_question"] for decision in shown["decisions"][:3]] == [
            "outside_orientation", "comfortable_settle", expected_third,
        ]
        assert shown["history"][:2] == [
            ["outside_orientation", pattern[0]], ["comfortable_settle", 0],
        ]
        assert shown["correct"] is True


def test_model_roundtrip_determinism_and_strict_integrity(fitted_models):
    tmp_path, model, _, examples = fitted_models
    path = tmp_path / "model.json"
    task.save_model(model, path)
    loaded = task.load_model(path)
    assert loaded == model
    repeated = task.fit_preference(examples, model.provenance)
    assert repeated.model_sha256 == model.model_sha256
    document = json.loads(path.read_text())
    document["weights"]["outside_orientation"]["bias"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        task.load_model(path)


def test_unsupported_history_explicitly_abstains(fitted_models):
    _, model, _, _ = fitted_models
    altered = task.PreferenceModel(
        model.version, model.memoryless, model.weights,
        {key: value for key, value in model.feature_counts.items() if key != "qa:outside_orientation=1"},
        model.provenance, model.model_sha256,
    )
    question, detail = altered.choose(
        (("outside_orientation", 1),),
        tuple(q for q in task.QUESTION_ORDER if q != "outside_orientation"),
    )
    assert question is None
    assert detail["status"] == "unsupported_history_abstention"


def test_all_policies_share_api_budget_evidence_and_legality(fitted_models):
    _, model, memoryless, _ = fitted_models
    item = episode((1, 1, 0, 0, 0, 1), "possible_discomfort")
    for policy in protocol.POLICIES:
        result = benchmark.run_policy(item, policy, model, memoryless)
        assert result["question_cost"] <= task.MAX_QUESTIONS
        assert result["budget_adherent"] and result["no_repeats"] and result["legal"]
        assert result["unsupported_confidence"] == 0
        assert len(result["evidence_trace"]) == result["question_cost"]
        assert all(row["evidence"].startswith("Observed answer") for row in result["evidence_trace"])


def test_partitions_are_reproducible_pattern_disjoint_and_structurally_overlapping():
    partitions = protocol.generate_episodes()
    assert partitions == protocol.generate_episodes()
    assert partitions != protocol.generate_episodes(protocol.GENERATOR_SEED + 1)
    protocol.validate_partitions(partitions)
    patterns = {
        split: {tuple(row["answers"]) for row in rows}
        for split, rows in partitions.items()
    }
    assert {split: len(rows) for split, rows in patterns.items()} == protocol.COUNTS
    assert patterns["train"].isdisjoint(patterns["validation"] | patterns["test"])
    assert patterns["validation"].isdisjoint(patterns["test"])
    assert len({row["opening"] for row in [
        task.DogQuestionAPI(benchmark._episode(partitions[s][0])).observe() for s in protocol.COUNTS
    ]}) == 1


def test_prepare_train_freezes_without_executing_test(tmp_path):
    directory = tmp_path / "dog-dev"
    preparation = benchmark.prepare(directory)
    assert preparation["outcomes_executed"] is False
    assert not (directory / "model.json").exists()
    training = benchmark.train(directory)
    assert training["no_test_outcomes_executed"] is True
    assert not (directory / "test-results.jsonl").exists()
    freeze = benchmark.check_frozen(directory)
    assert freeze["test_outcomes_inspected"] is False
    assert set(freeze["source_sha256"]) == set(preparation["source_sha256"])
    with pytest.raises(ValueError, match="already"):
        benchmark.train(directory)


def _rehash(document):
    body = {key: value for key, value in document.items() if key != "model_sha256"}
    document["model_sha256"] = task.digest(body)
    return document


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra=True),
    lambda d: d.update(memoryless=1),
    lambda d: d["weights"].pop(task.QUESTION_ORDER[0]),
    lambda d: d["weights"][task.QUESTION_ORDER[0]].update({"unknown-feature": 0.5}),
    lambda d: d["weights"][task.QUESTION_ORDER[0]].update({"bias": True}),
    lambda d: d["weights"][task.QUESTION_ORDER[0]].update({"bias": 1.01}),
    lambda d: d["feature_counts"].update({"bias": 0}),
    lambda d: d["feature_counts"].pop("depth=3"),
    lambda d: d["provenance"].update({"fit_split": "validation"}),
    lambda d: d["provenance"].update({"training_example_count": 1}),
])
def test_self_consistently_hashed_invalid_models_are_rejected(
    fitted_models, tmp_path, mutation,
):
    _, model, _, _ = fitted_models
    document = json.loads(json.dumps(model.document()))
    mutation(document)
    path = tmp_path / "invalid-model.json"
    path.write_text(json.dumps(_rehash(document)), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid|Incompatible"):
        task.load_model(path)


def test_missing_corrupt_nonfinite_and_incompatible_models_are_explicit(fitted_models, tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        task.load_model(tmp_path / "missing.json")
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        task.load_model(corrupt)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"weight": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        task.load_model(nonfinite)
    _, model, _, _ = fitted_models
    incompatible = tmp_path / "incompatible.json"
    document = model.document()
    document["version"] = "dog-history-preference-v999"
    incompatible.write_text(json.dumps(_rehash(document)), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        task.load_model(incompatible)


@pytest.fixture(scope="module")
def development_freeze(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dog-history-development") / "experiment"
    prepared = benchmark.prepare(
        directory, protocol.GENERATOR_SEED + 1, development_only=True,
    )
    trained = benchmark.train(directory)
    return directory, prepared, trained


def test_development_only_lifecycle_and_demo_validation(development_freeze, capsys):
    directory, prepared, trained = development_freeze
    assert prepared["development_only"] is True
    assert prepared["counts"]["test"] == 0
    assert trained["no_test_outcomes_executed"] is True
    freeze = benchmark.check_frozen(directory)
    assert freeze["model_sha256"] == trained["model_sha256"]
    shown = benchmark.demo(directory, "100100", "learned_history")
    assert shown["policy"] == "learned_history"
    assert shown["model_artifacts"]["history_model_sha256"] == trained["model_sha256"]
    assert shown["model_artifacts"]["fit_provenance"]["fit_split"] == "train"
    assert "not calibrated correctness" in shown["interpretation"]["support"]
    benchmark.main([
        "demo", str(directory), "--pattern", "011100", "--policy", "learned_history",
    ])
    captured = capsys.readouterr()
    assert json.loads(captured.out)["model_artifacts"]["selected_ranker_sha256"] == trained["model_sha256"]
    assert "zero does not mean zero wrong supported answers" in captured.err
    continuation = benchmark.continuation_demo(
        directory, (("comfortable_settle", 1),),
    )
    assert continuation["starter_source"].startswith("manually supplied")
    assert continuation["next_question"] == "recent_return"
    assert continuation["classification"].endswith("not a benchmark policy result")
    with pytest.raises(ValueError, match="Development-only"):
        benchmark.evaluate(directory)
    assert not (directory / "official-test-started.json").exists()


@pytest.mark.parametrize("mutation", ["extra", "missing", "modified"])
def test_exact_archived_source_inventory_and_bytes(development_freeze, tmp_path, mutation):
    source, _, _ = development_freeze
    directory = tmp_path / mutation
    shutil.copytree(source, directory)
    archived = directory / "sources" / "src" / "intuition_prototype" / "dog_history.py"
    if mutation == "extra":
        (directory / "sources" / "extra.py").write_text("# extra\n", encoding="utf-8")
    elif mutation == "missing":
        archived.unlink()
    else:
        archived.write_bytes(archived.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="archive|Archived"):
        benchmark.check_prepared(directory, execution=False)


def test_unsafe_source_archive_path_is_rejected(tmp_path, monkeypatch):
    source = Path(task.__file__)
    monkeypatch.setattr(benchmark, "source_paths", lambda: {"../escape.py": source})
    with pytest.raises(ValueError, match="Unsafe executable source path"):
        benchmark.prepare(
            tmp_path / "unsafe", protocol.GENERATOR_SEED + 2, development_only=True,
        )
    assert not (tmp_path / "escape.py").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_archive_demo_dispatch_is_not_recursive(fitted_models, tmp_path, monkeypatch, legacy):
    _, model, memoryless, _ = fitted_models
    directory = tmp_path / "completed-fixture"
    package = directory / "sources" / "src" / "intuition_prototype"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    entry = "demo" if legacy else "_live_demo"
    code = (
        "import json\n"
        "def read_json(path):\n"
        "    return json.loads(path.read_text())\n"
        f"def {entry}(directory, pattern, policy):\n"
        "    return {'policy': policy, 'archive_entry': 'direct'}\n"
    )
    if not legacy:
        code += (
            "def demo(*args):\n"
            "    raise AssertionError('completed demo recursively dispatched')\n"
        )
    (package / "dog_history_benchmark.py").write_text(code, encoding="utf-8")
    (directory / "preparation.json").write_text(
        json.dumps({"source_sha256": {}, "environment": {}}), encoding="utf-8",
    )
    (directory / "evaluation.json").write_text("{}", encoding="utf-8")
    task.save_model(model, directory / "model.json")
    task.save_model(memoryless, directory / "memoryless-model.json")
    monkeypatch.setattr(benchmark, "_verify_completed_transport", lambda path: ({
        "model_sha256": model.model_sha256,
        "memoryless_model_sha256": memoryless.model_sha256,
    }, None, None))
    result = benchmark._historical_demo(directory, "011100", "learned_history")
    assert result["archive_entry"] == "direct"
    assert result["classification"].startswith("verified archived-code")


@pytest.fixture(scope="module")
def archived_official(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / ".runtime" / "dog-history-v1"
    if not source.is_dir():
        pytest.skip(
            "Optional historical replay requires local .runtime/dog-history-v1; "
            "frozen experiment artifacts are deliberately not published."
        )
    directory = tmp_path_factory.mktemp("dog-history-historical") / "dog-history-v1"
    shutil.copytree(source, directory)
    return directory


def test_historical_report_is_read_only_and_uses_archived_code(
    archived_official, monkeypatch, capsys,
):
    directory = archived_official
    before = {
        path.relative_to(directory): benchmark.file_digest(path)
        for path in directory.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(benchmark, "source_fingerprints", lambda: {"live": "drift"})
    monkeypatch.setattr(benchmark, "environment", lambda: {"live": "drift"})
    monkeypatch.setattr(
        benchmark, "render_report",
        lambda *args: pytest.fail("historical report used live renderer"),
    )
    with pytest.raises(ValueError, match="source drift"):
        benchmark.check_frozen(directory, execution=True)
    monkeypatch.setattr(protocol, "VERSION", "future-version")
    text = benchmark.report(directory)
    assert text == (directory / "report.md").read_text(encoding="utf-8")
    benchmark.main(["report", str(directory)])
    captured = capsys.readouterr()
    assert captured.out == text
    assert "evaluator-only hindsight" in captured.err
    after = {
        path.relative_to(directory): benchmark.file_digest(path)
        for path in directory.rglob("*") if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize(("pattern", "earlier_answer", "expected_third"), [
    ("011100", "no", "discomfort_indicator"),
    ("101101", "yes", "outside_cue"),
])
def test_real_historical_demo_cli_uses_archived_v1(
    archived_official, capsys, pattern, earlier_answer, expected_third,
):
    benchmark.main([
        "demo", str(archived_official), "--pattern", pattern, "--policy", "learned_history",
    ])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["classification"].startswith("verified archived-code historical demonstration")
    assert result["execution_source"] == "frozen archived v1 source"
    assert result["model_artifacts"]["history_model_sha256"] == (
        "9240127f45bf68abcd2d3650a553faeaaae70712c0b2dbe4b79251d20dade21c"
    )
    assert result["model_artifacts"]["fit_provenance"]["fit_split"] == "train"
    assert [decision["chosen_question"] for decision in result["decisions"][:3]] == [
        "outside_orientation", "comfortable_settle", expected_third,
    ]
    assert result["history"][0] == ["outside_orientation", int(earlier_answer == "yes")]
    assert result["history"][1] == ["comfortable_settle", 0]
    assert result["correct"] is True
    assert "not a calibrated-correctness guarantee" in captured.err


@pytest.mark.parametrize(("answer", "expected"), [
    ("no", "outside_orientation"),
    ("yes", "recent_return"),
])
def test_real_historical_continue_cli_uses_archived_v1(
    archived_official, capsys, answer, expected,
):
    benchmark.main([
        "continue", str(archived_official), "--observed", f"comfortable_settle={answer}",
    ])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["starter_source"].startswith("manually supplied")
    assert result["next_question"] == expected
    assert result["model_sha256"] == (
        "9240127f45bf68abcd2d3650a553faeaaae70712c0b2dbe4b79251d20dade21c"
    )


def test_historical_report_rejects_extra_artifact(archived_official, tmp_path):
    directory = tmp_path / "extra-artifact"
    shutil.copytree(archived_official, directory)
    (directory / "unexpected.txt").write_text("not declared\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        benchmark.report(directory)


def test_self_consistently_resealed_invalid_raw_record_is_rejected(
    archived_official, tmp_path,
):
    directory = tmp_path / "invalid-raw"
    shutil.copytree(archived_official, directory)
    raw_path = directory / "test-results.jsonl"
    rows = benchmark._read_lines(raw_path)
    result = next(iter(rows[0]["policies"].values()))
    result["correct"] = not result["correct"] if result["correct"] is not None else True
    raw_path.unlink()
    benchmark._write_lines(raw_path, rows)
    evaluation_path = directory / "evaluation.json"
    evaluation = benchmark.read_json(evaluation_path)
    evaluation["test_result_sha256"] = benchmark.file_digest(raw_path)
    evaluation_path.unlink()
    benchmark.write_json(evaluation_path, evaluation)
    (directory / "evaluation.json.sha256").write_text(
        benchmark.file_digest(evaluation_path) + "\n", encoding="ascii",
    )
    with pytest.raises(ValueError, match="answer/latent-label|abstention"):
        benchmark.report(directory)
