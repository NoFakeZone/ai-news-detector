# ai-news-detector

Exploring and validating news on the internet through text feature extraction.

Features are based on the wiki article: [text features](https://github.com/NoFakeZone/ai-news-detector/wiki/text_features).

## Status

Implemented:

- **Punctuation extraction** — three variants behind a factory:
  - raw count
  - count per word (count / total words)
  - count per letter (count / total letters)

Planned:

- Part-of-speech (POS) tagging

## Project layout

```
ai-news-detector/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── ai_news_detector/
│       └── features/
│           ├── base.py            # TextFeatureExtractor ABC
│           ├── text_utils.py      # count_words, count_letters
│           └── punctuation/
│               ├── variants.py    # PunctuationVariant StrEnum
│               ├── base.py        # PunctuationExtractor ABC (holds punctuation_chars)
│               ├── count.py       # PunctuationCountExtractor (base type)
│               ├── ratio.py       # PerWord + PerLetter extractors
│               └── factory.py     # PunctuationExtractorFactory
└── tests/
    ├── test_text_utils.py
    ├── test_punctuation_count.py
    ├── test_punctuation_ratio.py
    └── test_punctuation_factory.py
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
| `scripts/run.sh` | Run `scripts/demo.py`, which prints each punctuation variant's output on a sample text. |
| `scripts/_activate.sh` | Shared helper sourced by the other scripts; creates venv if missing and activates it. Not meant to be run directly. |

## Usage

### Via the factory (recommended)

```python
from ai_news_detector.features.punctuation import (
    PunctuationExtractorFactory,
    PunctuationVariant,
)

text = "Hello, world! How are you?"

# Using the enum
counter = PunctuationExtractorFactory.create(PunctuationVariant.COUNT)
print(counter.extract(text))        # 3.0

# Using a string variant
per_word = PunctuationExtractorFactory.create("per_word")
print(per_word.extract(text))       # 3.0 / 5  = 0.6

per_letter = PunctuationExtractorFactory.create("per_letter")
print(per_letter.extract(text))     # 3.0 / 20 = 0.15
```

### Direct instantiation

```python
from ai_news_detector.features.punctuation import (
    PunctuationCountExtractor,
    PunctuationPerWordExtractor,
    PunctuationPerLetterExtractor,
)

PunctuationCountExtractor().extract("Hi!")          # 1.0
PunctuationPerWordExtractor().extract("hi, there!") # 1.0
PunctuationPerLetterExtractor().extract("Hi!")      # 0.5
```

### Custom punctuation set

The default punctuation set is `string.punctuation` (ASCII). Override it for Unicode punctuation (e.g. `—`, `„`, `…`):

```python
import string
from ai_news_detector.features.punctuation import PunctuationCountExtractor

chars = frozenset(string.punctuation) | {"—", "„", "…"}
extractor = PunctuationCountExtractor(punctuation_chars=chars)
extractor.extract("She said „hello"… — really?")
```

## Design

Extractors implement a shared abstract contract:

```python
class TextFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> float: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
```

Ratio extractors **compose** `PunctuationCountExtractor` rather than duplicating counting logic — `counter` is injectable via the constructor, which makes testing straightforward (a `MagicMock` can be passed in). Edge cases (empty text, punctuation-only text) return `0.0` instead of raising `ZeroDivisionError`.

POS tagging will follow the same pattern as a sibling `features/pos/` package.

## Adding a new text feature

All features follow the same pattern. Use this checklist when adding a new one (e.g. POS tagging, average word length, type-token ratio).

### 1. Create a package under `src/ai_news_detector/features/`

```
features/
└── your_feature/
    ├── __init__.py        # re-export public names
    ├── variants.py        # (optional) StrEnum of variants
    ├── base.py            # (optional) shared ABC holding common config
    ├── <variant>.py       # concrete extractor(s), one file per variant group
    └── factory.py         # factory mapping variants to classes
```

### 2. Extend `TextFeatureExtractor`

Every extractor must implement the root contract in `features/base.py`:

```python
from ai_news_detector.features.base import TextFeatureExtractor

@dataclass(frozen=True)
class MyFeatureExtractor(TextFeatureExtractor):
    @property
    def name(self) -> str:
        return "my_feature"

    def extract(self, text: str) -> float:
        ...
```

Guidelines:

- Return type is always `float` (uniform API across all features).
- Use `@dataclass(frozen=True)` — extractors must be immutable and hashable.
- The `name` property must return a stable, snake_case identifier used as a feature key downstream.
- Handle edge cases explicitly: empty text, whitespace-only, missing tokens. Never raise `ZeroDivisionError` — return `0.0` instead.

### 3. Prefer composition for derived variants

If a variant is a transformation of a base count (ratio, normalized value, log-scaled), **inject the base extractor as a field** — do not duplicate counting logic. See `features/punctuation/ratio.py` for the reference pattern:

```python
@dataclass(frozen=True)
class MyRatioExtractor(TextFeatureExtractor):
    counter: MyCountExtractor = field(default_factory=MyCountExtractor)
    ...
```

This keeps counting logic in one place and lets tests inject a `MagicMock` for the inner extractor.

### 4. Reuse `text_utils`

Common text helpers live in `features/text_utils.py` (`count_words`, `count_letters`). Add new shared helpers there rather than duplicating them inside feature packages.

### 5. Add a factory

Use a `StrEnum` for variants and a classmethod registry. The factory must accept both enum values and plain strings, and raise `ValueError` on unknown variants (free via `EnumClass(variant)`):

```python
class MyFeatureFactory:
    _registry: dict[MyVariant, type[TextFeatureExtractor]] = {
        MyVariant.A: MyAExtractor,
        MyVariant.B: MyBExtractor,
    }

    @classmethod
    def create(cls, variant: MyVariant | str) -> TextFeatureExtractor:
        return cls._registry[MyVariant(variant)]()
```

### 6. Re-export the public API

In `features/your_feature/__init__.py`, re-export the enum, extractors, and factory so callers can import from the package root:

```python
from ai_news_detector.features.your_feature.variants import MyVariant
from ai_news_detector.features.your_feature.factory import MyFeatureFactory
# ... etc.

__all__ = ["MyVariant", "MyFeatureFactory", ...]
```

### 7. Write tests under `tests/`

One file per module (`test_<feature>_<variant>.py`, `test_<feature>_factory.py`). Each extractor needs:

- A happy-path case with a known expected output.
- Edge cases: empty string, whitespace-only, inputs that would cause division by zero.
- A `name` property assertion.
- For composed extractors: a `MagicMock` delegation test proving the inner extractor is called (composition over duplication).
- For the factory: parametrized tests over every enum value, a string-input case, and a `ValueError` case for an unknown variant.

Use `pytest.approx` for float comparisons and `@pytest.mark.parametrize` to keep tests compact.

### 8. Run the suite

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
