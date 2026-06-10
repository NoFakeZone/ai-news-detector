import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import gc

# Make sure these match your actual imports
from feature_bert import MultiModalBertModel 
from dataset import NewsPopularityDataset
from load_dataset import load_dataset

# --- KONFIGURACJA LOGOWANIA ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- GLOBALNE PARAMETRY ---
BERT_MODEL_NAME = "allegro/herbert-base-cased"
NUM_CLASSES = 1 
BATCH_SIZE = 16
DATA_PATH = r'C:\Users\PC\OneDrive\Pulpit\projekty\ai-news-generator'
WIKIPEDIA_DICT_PATH = "wiki_popularity_dict.json"

# ==========================================
# --- LISTA KONFIGURACJI DO PRZETESTOWANIA ---
# ==========================================
TEST_CONFIGURATIONS = [
    {
        "output_dir": "wb_trainings/gemini-2.5-flash_wb_f_new_arch",
        "test_data": "gemini-2.5-flash",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
    {
        "output_dir": "wb_trainings/gemini-3-flash-previewwb_f_new_arch",
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
        {
        "output_dir": "wb_trainings/gpt-oss-120b_f_new_arch",
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
    {
        "output_dir": "wb_trainings\llama-3.3-70b-instruct-fp8-fast_f_new_arch",
        "test_data": "llama-3.3-70b-instruct-fp8-fast",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
    {
        "output_dir": r"wb_trainings\nemotron-3-120b-a12b_f_new_arch",
        "test_data": "nemotron-3-120b-a12b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": False,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
    {
        "output_dir": r"wb_trainings\gpt-oss-120b",
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
    {
        "output_dir": r"wb_trainings\gpt-oss-120b",
        "test_data": "gpt-oss-120b",
        "flags": {
            "USE_STYLISTIC_FEATURES": True,
            "BASIC_POPULARITY_INDEX": True,
            "WIKIPEDIA_POPULARITY_INDEX": False,
            "NKJP_POPULARITY_INDEX": False,
            "NORMALIZE_NKJP": False
        }
    },
]

def evaluate_model(config, device):
    output_dir = config["output_dir"]
    test_data = config["test_data"]
    flags = config["flags"]
    
    model_path = os.path.join(output_dir, "best_bert_stylistic_model.pt")
    
    logger.info(f"Loading test data '{test_data}'...")
    data = load_dataset(
        test_data, 
        DATA_PATH, 
        flags["USE_STYLISTIC_FEATURES"], 
        flags["BASIC_POPULARITY_INDEX"],
        flags["WIKIPEDIA_POPULARITY_INDEX"], 
        flags["NKJP_POPULARITY_INDEX"], 
        flags["NORMALIZE_NKJP"], 
        WIKIPEDIA_DICT_PATH
    )
    
    test_text = list(data[0])
    test_labels = torch.tensor(data[1])
    test_features = torch.tensor(data[2])

    logger.info(f"Test samples loaded: {len(test_text)}")
    logger.info(f"Loading model from: {model_path}")
    
    vector_size = test_features.shape[1] if test_features.numel() > 0 else 0
    
    model = MultiModalBertModel( 
        bert_model_name=BERT_MODEL_NAME, 
        vector_input_size=vector_size, 
        num_classes=NUM_CLASSES
    )
    
    # Wczytywanie wag
    model_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)
    model.eval()

    # Parametry normalizacji
    min_pop = model_dict['min_popularity_index']
    max_pop = model_dict['max_popularity_index']
    min_popularity_index_test = min_pop.to('cpu') if isinstance(min_pop, torch.Tensor) else min_pop
    max_popularity_index_test = max_pop.to('cpu') if isinstance(max_pop, torch.Tensor) else max_pop

    test_dataset = NewsPopularityDataset(
        test_text, 
        test_features, 
        test_labels, 
        BERT_MODEL_NAME, 
        use_features=flags["USE_STYLISTIC_FEATURES"], 
        min_popularity_index=min_popularity_index_test, 
        max_popularity_index=max_popularity_index_test
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_sum_target = 0.0
    
    test_tp = 0 
    test_fp = 0 
    test_fn = 0 

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            float_vectors = batch['float_vectors'].to(device)
            labels = batch['labels'].to(device).float().view(-1) 
            
            logits = model(input_ids, attention_mask, float_vectors).view(-1)
            loss = criterion(logits, labels)
            
            test_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            test_sum_target += labels.sum().item()

            test_tp += ((preds == 1) & (labels == 1)).sum().item()
            test_fp += ((preds == 1) & (labels == 0)).sum().item()
            test_fn += ((preds == 0) & (labels == 1)).sum().item()

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Testing in progress... Batch {batch_idx+1}/{len(test_loader)}")

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_avg_target = test_sum_target / test_total

    precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
    recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f">>> RESULTS FOR: {test_data} | MODEL: {output_dir} <<<")
    logger.info(f"TEST -> Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.2%} | Avg Target: {test_avg_target:.3f}")
    logger.info(f"METRICS -> Precision: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1_score:.2%}")
    logger.info("-" * 60)

    # Czyszczenie pamięci po każdej iteracji
    del model, test_loader, test_dataset, data, test_text, test_labels, test_features, model_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Chosen DEVICE: {device}')
    logger.info(f"Found {len(TEST_CONFIGURATIONS)} configurations to evaluate.\n")

    for idx, config in enumerate(TEST_CONFIGURATIONS):
        logger.info(f"========== RUNNING CONFIGURATION {idx + 1}/{len(TEST_CONFIGURATIONS)} ==========")
        logger.info(f"Output Dir: {config['output_dir']}")
        logger.info(f"Test Data: {config['test_data']}")
        
        try:
            evaluate_model(config, device)
        except Exception as e:
            logger.error(f"Error during evaluation of config {idx + 1}: {e}")
            logger.info("Skipping to next configuration...\n")

    logger.info("All evaluations completed successfully!")

if __name__ == "__main__":
    main()