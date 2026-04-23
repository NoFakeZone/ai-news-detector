# ai-news-detector

Exploring and validating news on the internet through text feature extraction.

Features are based on the wiki article: [text features](https://github.com/NoFakeZone/ai-news-detector/wiki/text_features).

## Status

Implemented:

- **Punctuation** — `punctuation_count`, `punctuation_per_word`, `punctuation_per_letter`
- **Part-of-speech (POS) tagging** — `pos_count`, `pos_per_word` (Polish, via spaCy `pl_core_news_sm`)
- **Text statistics** — `ttr`, `ttr_lemmatized`, `capital_ratio`, `avg_sentence_len`

## Project layout

```
ai-news-detector/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── ai_news_detector/
│       └── features/
│           ├── text_utils.py      # count_words, count_letters
│           ├── punctuation.py     # punctuation_count, punctuation_per_word, punctuation_per_letter
│           ├── pos.py             # pos_count, pos_per_word (Polish spaCy)
│           └── text_stats.py      # ttr, ttr_lemmatized, capital_ratio, avg_sentence_len
└── tests/
    ├── test_text_utils.py
    ├── test_punctuation.py
    ├── test_pos.py
    └── test_text_stats.py
```

## Requirements

- Python **3.11+** (uses `StrEnum` and modern typing)
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
| `scripts/run.sh` | Run `scripts/demo.py`, which demos punctuation, POS, and text stats features on sample texts. |
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
punctuation_count(text)        # 3.0
punctuation_per_word(text)     # 0.6  (3 / 5 words)
punctuation_per_letter(text)   # ≈ 0.1579 (3 / 19 letters)
```

Custom punctuation set (default is `string.punctuation`):

```python
import string
chars = frozenset(string.punctuation) | {"—", "„", "…"}
punctuation_count('She said „hello"… — really?', chars=chars)
```

### POS tagging (Polish)

```python
from ai_news_detector.features.pos import pos_count, pos_per_word

text = "Kot biegnie szybko przez zielony ogród."
pos_count(text, tag="NOUN")     # float — number of nouns
pos_per_word(text, tag="VERB")  # float — verbs / total words
```

Requires the `pl_core_news_sm` spaCy model (see [Setup](#setup)). An injectable `tagger` callable can be passed for testing.

### Text statistics

```python
from ai_news_detector.features.text_stats import (
    ttr,
    ttr_lemmatized,
    capital_ratio,
    avg_sentence_len,
)

text = "Kot biegnie. Pies biegnie."
ttr(text)              # type-token ratio (simple, regex-based)
ttr_lemmatized(text)   # TTR after lemmatisation via spaCy (requires pl_core_news_sm)
capital_ratio(text)    # ratio of non-sentence-start capitals to text length
avg_sentence_len(text) # average word count per sentence
```

An injectable `lemmatize` callable can be passed to `ttr_lemmatized` for testing.

## Design

All features are **plain functions** that return `float`. There are no classes, factories, or enums.

Each module is self-contained:

- `punctuation.py` — pure string operations, no dependencies beyond `text_utils`
- `pos.py` — spaCy loaded lazily via `functools.cache`; `tagger` is an injectable callable (`(str) -> list[tuple[str, str]]`) so tests never need the model
- `text_stats.py` — same pattern; `lemmatize` is an injectable callable (`(str) -> list[str]`) for `ttr_lemmatized`

**Edge-case convention.** Empty or whitespace-only text and division-by-zero inputs always return `0.0`.

## Adding a new text feature

### 1. Create a module under `src/ai_news_detector/features/`

Add a single file `features/your_feature.py`. No packages, no factories, no base classes.

### 2. Write plain functions returning `float`

```python
# features/your_feature.py

def your_feature(text: str) -> float:
    if not text.strip():
        return 0.0
    ...
```

Guidelines:

- Return type is always `float`.
- Handle edge cases explicitly: empty text, whitespace-only, potential division by zero — always return `0.0` instead of raising.
- If the function depends on an external model (e.g. spaCy), load it lazily with `functools.cache` and accept an injectable callable parameter so tests never need the model installed.
- Shared text helpers belong in `features/text_utils.py` (`count_words`, `count_letters`).

### 3. Write tests in `tests/test_<feature>.py`

```python
from ai_news_detector.features.your_feature import your_feature

def test_empty():
    assert your_feature("") == 0.0

def test_happy_path():
    assert your_feature("some text") == pytest.approx(expected)
```

Each function needs:

- A happy-path case with a known expected output.
- Edge cases: empty string, whitespace-only, division-by-zero inputs.
- For functions with injectable callables: a test that confirms the callable is invoked with the right argument (use a plain lambda or closure — no `MagicMock` needed).
- Integration tests that hit a real external model should use `pytest.mark.slow` and `pytest.mark.skipif` guarded by a model-availability check.

Use `pytest.approx` for float comparisons and `@pytest.mark.parametrize` to keep tests compact.

### 4. Run the suite

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
