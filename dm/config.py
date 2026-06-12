from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DM_DIR = REPO_ROOT / "dm"

MIN_WORDS = 15
AI_THRESHOLD = 0.5
MODEL_PATH = DM_DIR / "best_model.pt"
BERT_MODEL_NAME = "allegro/herbert-base-cased"
NKJP_DICT_PATH = REPO_ROOT / "nkjp_popularity_dict.json"
