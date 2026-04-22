# ai-news-detector

Exploring and validating news on the internet through text feature extraction.

Features are based on the wiki article: [text features](https://github.com/NoFakeZone/ai-news-detector/wiki/text_features).

## Status

Implemented:

- **Punctuation extraction** — three plain functions:
  - `punctuation_count` — raw count
  - `punctuation_per_word` — count / total words
  - `punctuation_per_letter` — count / total letters
- **POS tagging** — `pos_count` and `pos_per_word` functions accepting a UD tag string (`"NOUN"`, `"VERB"`, `"ADJ"`, …); uses [spaCy](https://spacy.io/) with the `pl_core_news_sm` Polish model (install with `python -m spacy download pl_core_news_sm`)

## Project layout

```
ai-news-detector/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── ai_news_detector/
│       └── features/
│           ├── __init__.py         # re-exports all public functions
│           ├── text_utils.py       # count_words, count_letters
│           ├── punctuation.py      # punctuation_count / per_word / per_letter
│           └── pos.py              # pos_count, pos_per_word, default_tagger
└── tests/
    ├── test_text_utils.py
    ├── test_punctuation_count.py
    ├── test_punctuation_ratio.py
    ├── test_pos_count.py
    ├── test_pos_ratio.py
    └── test_pos_tagger.py
```

## Requirements

- Python **3.11+**
- `pip`

## Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/NoFakeZone/ai-news-detector.git
cd ai-news-detector
```

### 2. Bootstrap the environment (recommended)

The `scripts/` folder contains bash helpers that handle venv creation, activation, installing deps, running the demo, running tests, and building. Run them from the repo root (on Windows, use Git Bash):

```bash
bash scripts/setup.sh
```

This creates `venv/`, activates it, upgrades `pip/setuptools/wheel`, and installs the package in editable mode with dev dependencies. Re-running is safe — an existing venv is reused.

After setup, activate the venv in your own shell so subsequent `python`/`pytest` calls resolve to the venv interpreter:

```bash
source venv/Scripts/activate   # Windows (Git Bash)
source venv/bin/activate       # Linux / macOS
```

### 3. Manual setup (alternative)

If you prefer not to use the helper scripts:

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Windows (cmd):

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Pip-only workflow using `requirements.txt`:

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Run the tests

```bash
bash scripts/test.sh      # via helper (activates venv for you)
pytest -v                 # or directly, if venv is already active
```

## Scripts

All scripts live under `scripts/` and can be invoked from any working directory (they `cd` to the repo root). Each one activates `venv/` before doing anything.

| Script | What it does |
|---|---|
| `scripts/setup.sh` | First-time bootstrap: create venv, activate, install `-e ".[dev]"`. |
| `scripts/install.sh` | Reinstall the package with dev deps into the existing venv. |
| `scripts/test.sh` | Run `pytest -v`. Extra args forward to pytest (`scripts/test.sh -k per_word`). |
| `scripts/build.sh` | Build sdist + wheel into `dist/` (installs `build` on demand). |
| `scripts/run.sh` | Run `scripts/demo.py`, which prints each punctuation and POS variant on a sample text. |
| `scripts/_activate.sh` | Shared helper sourced by the other scripts; creates venv if missing and activates it. Not meant to be run directly. |

## Usage

### Punctuation

```python
from ai_news_detector.features.punctuation import (
    punctuation_count,
    punctuation_per_word,
    punctuation_per_letter,
)

text = "Hello, world! How are you?"

punctuation_count(text)       # 3.0
punctuation_per_word(text)    # 3.0 / 5  = 0.6
punctuation_per_letter(text)  # 3.0 / 19 ≈ 0.1579
```

### POS tagging

```python
from ai_news_detector.features.pos import pos_count, pos_per_word

text = "Kot biegnie szybko przez zielony ogród."

pos_count(text, tag="NOUN")       # number of nouns as a float
pos_per_word(text, tag="VERB")    # verbs per word
```

The `tag` argument is any [Universal Dependencies POS tag](https://universaldependencies.org/u/pos/) string. Default is `"NOUN"`.

### Custom punctuation set

The default punctuation set is `string.punctuation` (ASCII). Override it for Unicode punctuation (e.g. `—`, `„`, `…`) via the `chars` keyword:

```python
import string
from ai_news_detector.features.punctuation import punctuation_count

chars = frozenset(string.punctuation) | {"—", "„", "…"}
punctuation_count("She said „hello"… — really?", chars=chars)
```

### Custom tagger (testing or non-spaCy backends)

`pos_count` and `pos_per_word` accept a `tagger` callable so the spaCy boundary can be swapped out (e.g. for tests or alternative NLP backends). It must take a string and return a `list[tuple[str, str]]` of `(token, pos_tag)` pairs:

```python
from ai_news_detector.features.pos import pos_count

def fake_tagger(text):
    return [("Kot", "NOUN"), ("biegnie", "VERB")]

pos_count("Kot biegnie", tag="NOUN", tagger=fake_tagger)  # 1.0
```

## Design

Each feature variant is a plain module-level function returning `float`. There are no classes, ABCs, or factory dispatch layers — callers import the function they want and call it.

Conventions every function follows:

- **Return type is always `float`** (uniform API across all features).
- **Edge cases return `0.0`** — empty text, whitespace-only, or anything that would divide by zero. Never raise `ZeroDivisionError`.
- **Derived variants delegate to the base count** rather than duplicating logic. `punctuation_per_word` calls `punctuation_count` internally; `pos_per_word` calls `pos_count` internally.
- **External dependencies are injectable as parameters** with sensible defaults — `chars` for punctuation, `tagger` for POS — so tests can pass mocks without monkey-patching.

Shared text-tokenization helpers live in `features/text_utils.py` (`count_words`, `count_letters`).

## Adding a new text feature

The pattern is intentionally minimal. To add a feature (e.g. average word length, type-token ratio):

### 1. Create one module under `src/ai_news_detector/features/`

```
features/
└── your_feature.py
```

A single flat module. Don't create a package directory unless the feature genuinely needs multiple files.

### 2. Write one function per variant

```python
from ai_news_detector.features.text_utils import count_words

def my_feature_count(text: str) -> float:
    ...

def my_feature_per_word(text: str) -> float:
    words = count_words(text)
    if words == 0:
        return 0.0
    return my_feature_count(text) / words
```

Guidelines:

- Return `float`.
- Handle empty / whitespace-only / divide-by-zero by returning `0.0`.
- Derived variants should call the base count function directly, not duplicate logic.
- Make external dependencies (taggers, models, char sets) optional parameters with defaults, so tests can inject mocks.

### 3. Reuse `text_utils`

Tokenization helpers (`count_words`, `count_letters`) live in `features/text_utils.py`. Add new shared helpers there rather than duplicating them.

### 4. Re-export from `features/__init__.py`

Add the new functions to the package-level imports and `__all__` so callers can write `from ai_news_detector.features import my_feature_count`.

### 5. Write tests under `tests/`

One file per variant group: `test_<feature>_<variant>.py`. Each function needs:

- Parametrized happy-path cases with known expected outputs.
- Edge cases: empty string, whitespace-only, divide-by-zero inputs.
- For functions taking a `tagger` (or other injectable dep): assertions that the dep is called with the right args, and that it is **not** called when the early-return short-circuits.

Use `pytest.approx` for float comparisons and `@pytest.mark.parametrize` to keep tests compact.

### 6. Run the suite

```bash
pytest -v
```

All existing and new tests should pass before opening a PR.

## Code style

This repo ships an `.editorconfig`. Most editors and IDEs pick it up automatically; for VS Code install the *EditorConfig for VS Code* extension. Key rules:

- **4-space indentation for Python** (never tabs)
- 2-space indentation for TOML, YAML, JSON, Markdown
- LF line endings (CRLF only for `.bat` / `.cmd`)
- UTF-8, trailing whitespace trimmed, final newline inserted
