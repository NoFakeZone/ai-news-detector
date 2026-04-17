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

Python **3.11+** is required (uses `StrEnum` and modern typing). Tests are configured in `pyproject.toml` with `testpaths = ["tests"]` and `pythonpath = ["src"]` — no `PYTHONPATH` setup needed when running pytest. The `run.sh` demo sets `PYTHONPATH=src` itself so it works without an editable install.

## Architecture

This is a text-feature extraction library for classifying news authenticity. Each feature lives under `src/ai_news_detector/features/<feature>/` and follows the same four-part shape: **variants enum → base ABC → concrete extractors → factory**. Currently implemented: `punctuation/` and `pos/`.

**Root contract** (`features/base.py`): every extractor is a `TextFeatureExtractor` subclass exposing `extract(text) -> float` and a `name` property (stable snake_case identifier used as a feature key). Return type is uniformly `float` across *all* features — this is non-negotiable.

**Composition over duplication for derived variants.** When a variant is a transformation of a base count (e.g. ratio, normalized, log-scaled), inject the base extractor as a dataclass field rather than re-implementing the count. `PunctuationPerWordExtractor` / `PunctuationPerLetterExtractor` in `punctuation/ratio.py` demonstrate the pattern: they hold a `counter: PunctuationCountExtractor` field and delegate. This keeps counting logic in one place and lets tests inject a `MagicMock` for the inner extractor.

**Immutability.** Extractors are `@dataclass(frozen=True)` — they must be hashable and safe to reuse. Per-feature shared configuration (like `punctuation_chars`) is held on a feature-level ABC (`punctuation/base.py`) so all variants inherit it consistently.

**Edge-case convention.** Empty text, whitespace-only text, and inputs that would otherwise divide by zero must return `0.0` — never raise `ZeroDivisionError`. Shared helpers for tokenization live in `features/text_utils.py` (`count_words`, `count_letters`); add new shared helpers there rather than duplicating inside feature packages.

**Factory pattern.** Each feature exposes a factory whose `_registry` maps a `StrEnum` variant to the concrete extractor class. `create()` accepts both the enum and a plain string (passing a string to `EnumClass(value)` auto-raises `ValueError` for unknown variants — rely on this instead of hand-rolled validation). Public names (enum, extractors, factory) are re-exported from each feature package's `__init__.py` so callers import from the package root.

## Adding a new feature

When adding a new feature, mirror the `punctuation/` or `pos/` package layout exactly and follow the checklist in `README.md` §"Adding a new text feature". Tests go under `tests/` as `test_<feature>_<variant>.py` and `test_<feature>_factory.py`; factory tests must be parametrized over every enum value and include a `ValueError` case for unknown variants.

## Code style

`.editorconfig` is authoritative: 4-space indent for Python, 2-space for TOML/YAML/JSON/Markdown, LF line endings, UTF-8, trailing whitespace trimmed.
