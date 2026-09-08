import pytest

from intuition_prototype import __version__
from intuition_prototype import app
from intuition_prototype.app import build_ui, environment_health
from intuition_prototype.config import Settings
from intuition_prototype.llm import create_adapter
from intuition_prototype.storage import count_entries, initialize, read_entry, write_entry


def test_sqlite_round_trip(tmp_path):
    database = tmp_path / "nested" / "test.sqlite3"
    initialize(database)
    content = "A demo's input; DROP TABLE demo_entries;"
    entry_id = write_entry(database, content)
    initialize(database)
    assert read_entry(database, entry_id) == content
    assert count_entries(database) == 1
    assert read_entry(database, entry_id + 1) is None


def test_imports_and_default_config():
    assert __version__ == "0.1.0"
    settings = Settings.from_env({})
    assert settings.mode == "demo"
    response = create_adapter(settings).generate("hello")
    assert "OFFLINE DEMO" in response
    assert "not an LLM response" in response
    assert response.endswith("hello")


@pytest.mark.parametrize("mode", ["remote", "invalid", ""])
def test_unsupported_mode_is_explicit(mode):
    with pytest.raises(ValueError, match="Only INTUITION_MODE=demo"):
        Settings.from_env({"INTUITION_MODE": mode})


def test_empty_database_path_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        Settings.from_env({"INTUITION_DB_PATH": " "})


def test_health_and_ui(tmp_path):
    database = tmp_path / "health.sqlite3"
    settings = Settings.from_env({"INTUITION_DB_PATH": str(database)})
    health = environment_health(settings)
    assert health["mode"] == "demo"
    assert health["stored_demo_entries"] == 0
    assert health["database"] == str(database.resolve())
    ui = build_ui(settings)
    assert ui.analytics_enabled is False
    ui.close()


def test_launch_is_local_only(monkeypatch):
    launched = {}

    class FakeUI:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(app, "build_ui", FakeUI)
    app.main()
    assert launched["server_name"] == "127.0.0.1"
    assert launched["server_port"] == 7860
    assert launched["share"] is False
    assert launched["enable_monitoring"] is False
    assert launched["inbrowser"] is False
    assert launched["run_history"] is False
    assert launched["footer_links"] == []
