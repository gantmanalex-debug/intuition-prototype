# Intuition environment scaffold

This is **environment setup only**, not the intuition research prototype.
It provides an importable Python package, standard-library SQLite storage,
pytest smoke tests, and a loopback-only Gradio health/demo interface.
No GPU, paid provider, credentials, model downloads, or LLM API calls are needed.

## Selected platform

- Windows host with existing WSL2 Ubuntu 24.04.3 LTS.
- Ubuntu Python 3.12.3 and SQLite 3.45.1.
- Verified with Gradio 6.26.0, pytest 9.1.1, and pip 26.2.1.
- Project: `C:\Users\AlexGantman\Projects\intuition-prototype`.
- Isolated **Linux** virtual environment: `.venv` in the project.
  Do not use this environment with Windows Python.

No Windows features, distributions, global Python packages, or global settings
were changed. Ubuntu lacks pip/ensurepip, so pip was bootstrapped into `.venv`
without sudo. The existing Windows `uv` is not needed for this WSL environment.

## Open, run, and test

From PowerShell, enter Ubuntu at the project directory:

```powershell
wsl -d Ubuntu --cd 'C:\Users\AlexGantman\Projects\intuition-prototype'
```

Then run these commands **inside Ubuntu**, not PowerShell:

```bash
source .venv/bin/activate
python -m pytest -q
python -m intuition_prototype.app
```

Open **http://127.0.0.1:7860** in the Windows browser. The app always binds to
`127.0.0.1`, uses `share=False`, and disables Gradio analytics and monitoring.
Stop a foreground server with Ctrl+C. Stop an already running server before
launching a second instance; an occupied port is reported rather than silently
switching ports. `intuition-ui` is an equivalent entry point after activation.
Browser run history is disabled, and the theme uses local system fonts.

In VS Code, use the WSL extension and **WSL: Reopen Folder in WSL** if available,
then select `.venv/bin/python` as the Python interpreter. Installing editor
extensions is optional and was not part of setup.

## Recreate dependencies

An internet connection is needed for installation, not for an LLM service.
With Python 3.12+ and working ensurepip, run inside Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip --isolated install -e '.[dev]'
python -m pytest -q
```

On this Ubuntu installation, ensurepip is absent. For a **new** environment,
use this no-admin alternative instead:

```bash
python3 -m venv --without-pip .venv
mkdir -p .runtime
curl --fail --location https://bootstrap.pypa.io/pip/pip.pyz --output .runtime/pip.pyz
.venv/bin/python .runtime/pip.pyz --isolated install pip
rm .runtime/pip.pyz
.venv/bin/python -m pip --isolated install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Dependencies are declared in `pyproject.toml`: Gradio >=6.26,<7 and the `dev` extra
with pytest. SQLite comes with Python. This modest scaffold does not pin a
complete transitive lockfile; package versions may change on a fresh install.
Installation on this Windows-mounted directory can be slow because WSL crosses
the filesystem boundary. No administrator installation is required.

## Configuration and data

The app reads process environment variables. `.env.example` documents them;
it is **not automatically loaded**, and no real `.env` or credentials are needed.

| Variable | Default | Meaning |
| --- | --- | --- |
| `INTUITION_MODE` | `demo` | Only supported mode; other values fail explicitly. |
| `INTUITION_DB_PATH` | `.runtime/intuition.sqlite3` | SQLite path, relative to the current working directory unless absolute. |

The UI initializes the SQLite schema and displays runtime versions, database
location, and the result of a database query. The demo simply echoes input with
an explicit **not an LLM response** label; inputs are not persisted.
The storage module has a small parameterized write/read API verified by tests.
It is not a research observation/hypothesis data model.

Local databases, runtime files, credentials, virtual environments, and caches
are excluded by `.gitignore`. This is a local development UI, not an authenticated
multi-user service. Do not expose it through a proxy, public tunnel, or LAN bind.

## Architecture boundary

- `config.py`: explicit validated demo configuration.
- `llm.py`: replaceable `LLMAdapter` protocol and deterministic `DemoAdapter`.
- `storage.py`: minimal SQLite initialization and demo entry persistence.
- `app.py`: environment health and local Gradio interface.
- `tests/test_smoke.py`: imports/configuration, SQLite initialize/write/read,
  health reporting, and UI construction.

Future research may add a unified controller choosing questions and investigating
through ask/test/reframe, distinguish observations from hypotheses, and later
introduce a learned selector and fuse. None of these speculative algorithms is
implemented here. There is no real-world tool executor.

A future remote adapter must explicitly implement the protocol, extend mode
validation and the factory, and define credentials, timeouts, error handling,
and user-approved cost controls. The current scaffold deliberately selects no
provider and rejects `remote` rather than silently falling back to demo.
