import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

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

# --- PARAMETRY (Muszą być identyczne jak podczas treningu!) ---
BERT_MODEL_NAME = "allegro/herbert-base-cased"
NUM_CLASSES = 1 
BATCH_SIZE = 16
TEST_DATA = 'gpt-oss-120b'
DATA_PATH = r'C:\Users\PC\OneDrive\Pulpit\projekty\ai-news-generator'

# Flagi cech - Ustaw na takie, na jakich trenowano model!
BASIC_POPULARITY_INDEX = False
WIKIPEDIA_POPULARITY_INDEX = False
USE_STYLISTIC_FEATURES = False

# Ścieżka do wytrenowanego modelu
MODEL_PATH = os.path.join("wb_training_run_nf_gpt-oss-120b", "best_bert_stylistic_model.pt")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Chosen DEVICE: {device}')

    # --- 1. PRZYGOTOWANIE DANYCH TESTOWYCH ---
    logger.info("Loading test data...")
    data = load_dataset(TEST_DATA, DATA_PATH, USE_STYLISTIC_FEATURES, BASIC_POPULARITY_INDEX, WIKIPEDIA_POPULARITY_INDEX)
    
    test_text = list(data[0])
    test_labels = torch.tensor(data[1])
    test_features = torch.tensor(data[2])

    logger.info(f"Test samples loaded: {len(test_text)}")

    test_dataset = NewsPopularityDataset(test_text, test_features, test_labels, BERT_MODEL_NAME, use_features=USE_STYLISTIC_FEATURES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. INICJALIZACJA I WCZYTANIE MODELU ---
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    vector_size = test_features.shape[1] if test_features.numel() > 0 else 0
    
    model = MultiModalBertModel( 
        bert_model_name=BERT_MODEL_NAME, 
        vector_input_size=vector_size, 
        num_classes=NUM_CLASSES
    )
    
    # map_location=device pozwala wczytać model wytrenowany na GPU na maszynę z samym CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # --- 3. PĘTLA TESTOWA ---
    logger.info("--- Starting Evaluation ---")
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_sum_target = 0.0

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

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Testing in progress... Batch {batch_idx+1}/{len(test_loader)}")

    # --- 4. PODSUMOWANIE WYNIKÓW ---
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_avg_target = test_sum_target / test_total

    logger.info(f">>> FINAL TEST RESULTS <<<")
    logger.info(f"TEST -> Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.2%} | Avg Target {test_avg_target:.3f}")

if __name__ == "__main__":
    main()