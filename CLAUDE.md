# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Helper scripts live under `scripts/` and are the preferred entrypoints — each one `cd`s to the repo root and activates `venv/` (creating it if missing) before running. On Windows, invoke via Git Bash.

| Script | Purpose |
|---|---|
| `bash scripts/setup.sh` | First-time bootstrap: create venv, activate, `pip install -e ".[dev]"`. |
| `bash scripts/install.sh` | Reinstall deps into existing venv (e.g. after `pyproject.toml` changes). |
| `bash scripts/test.sh [args]` | `pytest -v`, extra args forwarded (e.g. `bash scripts/test.sh -k ttr`). |
| `bash scripts/build.sh` | Build sdist + wheel into `dist/` (installs `build` on demand). |
| `bash scripts/run.sh` | Run `scripts/demo.py` exercising all feature functions on sample texts. |
| `scripts/_activate.sh` | Shared sourced helper — not meant to be executed directly. |

Running pytest directly (when venv is already active):

```bash
pytest -v                              # full suite
pytest tests/test_punctuation.py -v   # single file
pytest tests/test_pos.py::test_count  # single test
```

Python **3.11+** is required (uses `StrEnum` and modern typing). Tests are configured in `pyproject.toml` with `testpaths = ["tests"]` and `pythonpath = ["src"]` — no `PYTHONPATH` setup needed when running pytest.

## Architecture

This is a text-feature extraction library for classifying news authenticity. Features are **plain functions returning `float`** — no classes, factories, or enums.

Each feature lives in a single flat module under `src/ai_news_detector/features/`:

| Module | Functions |
|---|---|
| `punctuation.py` | `punctuation_count`, `punctuation_per_word`, `punctuation_per_letter` |
| `pos.py` | `pos_count`, `pos_per_word` |
| `text_stats.py` | `ttr`, `ttr_lemmatized`, `capital_ratio`, `avg_sentence_len` |
| `text_utils.py` | `count_words`, `count_letters` (shared helpers) |

**Return type is always `float`** across all features — non-negotiable.

**Edge-case convention.** Empty text, whitespace-only text, and inputs that would cause division by zero must return `0.0` — never raise.

**Injectable callables for external models.** `pos.py` accepts a `tagger: (str) -> list[tuple[str, str]]` parameter; `text_stats.py` accepts a `lemmatize: (str) -> list[str]` parameter for `ttr_lemmatized`. Both default to a spaCy-backed implementation that loads `pl_core_news_sm` lazily via `functools.cache`. Pass a plain lambda or closure in tests — the model is never required.

**Shared text helpers** live in `text_utils.py`. Add new helpers there rather than duplicating inside feature modules.

## Adding a new feature

1. Add a single file `src/ai_news_detector/features/your_feature.py` with plain functions.
2. If the feature needs an external model, load it lazily with `functools.cache` and accept an injectable callable parameter (see `pos.py` or `text_stats.py`).
3. Add `tests/test_your_feature.py` — one file, no sub-files. Required coverage:
   - Happy-path cases with known expected outputs.
   - Edge cases: empty string, whitespace-only, division-by-zero inputs.
   - For injectable callables: confirm the callable is invoked with the correct argument (plain lambda, no `MagicMock`).
   - spaCy integration tests must use `pytest.mark.slow` and `pytest.mark.skipif` guarded by a model-availability check.
4. Re-export from `features/__init__.py` if the function should be importable from the package root.
5. Run `pytest -v` — all tests must pass.

## Code style

`.editorconfig` is authoritative: 4-space indent for Python, 2-space for TOML/YAML/JSON/Markdown, LF line endings, UTF-8, trailing whitespace trimmed.
