# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Helper scripts live under `scripts/` and are the preferred entrypoints — each one `cd`s to the repo root and activates `venv/` (creating it if missing) before running. On Windows, invoke via Git Bash.

| Script | Purpose |
|---|---|
| `bash scripts/setup.sh` | First-time bootstrap: create venv, activate, `pip install -e ".[dev]"`. |
| `bash scripts/install.sh` | Reinstall deps into existing venv (e.g. after `pyproject.toml` changes). |
| `bash scripts/test.sh [args]` | `pytest -v`, extra args forwarded (e.g. `bash scripts/test.sh -k per_word`). |
| `bash scripts/build.sh` | Build sdist + wheel into `dist/` (installs `build` on demand). |
| `bash scripts/run.sh` | Run `scripts/demo.py` exercising all punctuation and POS variants. |
| `scripts/_activate.sh` | Shared sourced helper — not meant to be executed directly. |

Running pytest directly (when venv is already active):

```bash
pytest -v                                            # full suite
pytest tests/test_punctuation_ratio.py -v            # single file
pytest tests/test_punctuation_ratio.py::test_name    # single test
```

Python **3.11+** is required. Tests are configured in `pyproject.toml` with `testpaths = ["tests"]` and `pythonpath = ["src"]` — no `PYTHONPATH` setup needed when running pytest. The `run.sh` demo sets `PYTHONPATH=src` itself so it works without an editable install.

## Architecture

This is a text-feature extraction library for classifying news authenticity. Each feature lives in a single flat module under `src/ai_news_detector/features/` (`punctuation.py`, `pos.py`) and exposes one plain function per variant. There are no ABCs, dataclasses, enums, or factories — by deliberate choice. Currently implemented: `punctuation` (count, per_word, per_letter) and `pos` (count, per_word).

**Uniform return type.** Every feature function returns `float`. This is non-negotiable.

**Composition by direct call.** When a variant is a transformation of a base count (ratio, normalized, log-scaled), call the base function directly rather than re-implementing it. `punctuation_per_word` calls `punctuation_count`; `pos_per_word` calls `pos_count`. No injected "counter" object — just function calls.

**Edge-case convention.** Empty text, whitespace-only text, and inputs that would otherwise divide by zero must return `0.0` — never raise `ZeroDivisionError`. Shared tokenization helpers live in `features/text_utils.py` (`count_words`, `count_letters`); add new shared helpers there rather than duplicating inside feature modules.

**Injectable boundaries via parameters.** External dependencies are exposed as optional function parameters with defaults — `chars: frozenset[str] = DEFAULT_PUNCTUATION` for punctuation, `tagger: Callable = default_tagger` for POS. Tests pass `MagicMock` through these parameters; no monkey-patching needed. Don't introduce a class or Protocol for what a callable already expresses.

**No factories, no enums.** Callers import functions directly: `from ai_news_detector.features.pos import pos_count`. POS tags are plain UD strings (`"NOUN"`, `"VERB"`). Public names are re-exported from `features/__init__.py` so `from ai_news_detector.features import punctuation_per_word` also works.

## Adding a new feature

Mirror the existing `punctuation.py` / `pos.py` shape: one flat module, one function per variant, derived variants call base variants directly, external deps as optional parameters. Follow the checklist in `README.md` §"Adding a new text feature". Tests go under `tests/` as `test_<feature>_<variant>.py` and must cover happy paths (parametrized), edge cases (empty / whitespace / divide-by-zero), and — for functions with an injectable dep — both the called-with-right-args case and the not-called case for early returns.

## Code style

`.editorconfig` is authoritative: 4-space indent for Python, 2-space for TOML/YAML/JSON/Markdown, LF line endings, UTF-8, trailing whitespace trimmed.
