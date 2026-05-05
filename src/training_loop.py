import os
import random
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Make sure these match your actual imports
from feature_bert import MultiModalBertModel 
from dataset import NewsPopularityDataset
from load_dataset import load_dataset

# --- KONFIGURACJA ŚCIEŻEK I FOLDERÓW ---
OUTPUT_DIR = "wb_training_run_nf_gpt-oss-120b" # Nazwa folderu na logi i modele
os.makedirs(OUTPUT_DIR, exist_ok=True) # Tworzy folder, jeśli nie istnieje

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_bert_stylistic_model.pt")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "latest_checkpoint.pt")
LOG_PATH = os.path.join(OUTPUT_DIR, "training_log.txt")

# --- KONFIGURACJA LOGOWANIA ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH), # Zapis do pliku w folderze
        logging.StreamHandler()                  
    ]
)
logger = logging.getLogger(__name__)

# --- KONFIGURACJA I PARAMETRY ---
BERT_MODEL_NAME = "allegro/herbert-base-cased"
NUM_CLASSES = 1 
BATCH_SIZE = 16
REAL_BATCH = 16
BATCH_ACCUMULATION = int(BATCH_SIZE / REAL_BATCH)
LEARNING_RATE = 2e-5
EPOCHS = 30
WARMUP_PROPORTION = 0.1 
TEST_DATA = 'gpt-oss-120b'
DATA_PATH = r'C:\Users\PC\OneDrive\Pulpit\projekty\ai-news-generator'
BASIC_POPULARITY_INDEX = False
WIKIPEDIA_POPULARITY_INDEX = False
USE_STYLISTIC_FEATURES = False

RESUME_TRAINING = False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Chosen DEVICE: {device}')
logger.info(f"All outputs will be saved to directory: {OUTPUT_DIR}/")

# --- PRZYGOTOWANIE DANYCH ---
data = load_dataset(TEST_DATA, DATA_PATH, USE_STYLISTIC_FEATURES, BASIC_POPULARITY_INDEX, WIKIPEDIA_POPULARITY_INDEX)
size_of_train = len(data[3])
indices = random.sample(range(size_of_train), int(size_of_train * 0.1))
# ... (Zakładam, że Twoje dane są wczytywane poprawnie tak jak wcześniej) ...

# Poniżej znajduje się fragment z maskami - upewnij się, że data jest wczytane
test_text = list(data[0])
test_labels = torch.tensor(data[1])
test_features = torch.tensor(data[2])

train_text = np.array(data[3])
train_labels = torch.tensor(data[4])
train_features = torch.tensor(data[5])

val_text = train_text[indices]
val_labels = train_labels[indices]
val_features = train_features[indices]

mask = torch.ones(len(train_text), dtype=torch.bool)
mask[indices] = False

train_text = list(train_text[mask])
train_labels = train_labels[mask] 
train_features = train_features[mask]
val_text = list(val_text)

logger.info(f"Train samples: {len(train_text)} | Val samples: {len(val_text)} | Test samples: {len(test_text)}")

# --- INICJALIZACJA DATASETÓW I LOADERÓW ---
train_dataset = NewsPopularityDataset(train_text, train_features, train_labels, BERT_MODEL_NAME)
val_dataset = NewsPopularityDataset(val_text, val_features, val_labels, BERT_MODEL_NAME)
test_dataset = NewsPopularityDataset(test_text, test_features, test_labels, BERT_MODEL_NAME)

train_loader = DataLoader(train_dataset, batch_size=REAL_BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=REAL_BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=REAL_BATCH, shuffle=False)

# --- INICJALIZACJA MODELU ---
model = MultiModalBertModel( 
    bert_model_name=BERT_MODEL_NAME, 
    vector_input_size=train_features.shape[1], 
    num_classes=NUM_CLASSES
)
model.to(device)

# --- OPTYMALIZATOR, FUNKCJA STRATY I SCHEDULER ---
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

steps_per_epoch = math.ceil(len(train_loader) / BATCH_ACCUMULATION)
total_training_steps = steps_per_epoch * EPOCHS
num_warmup_steps = int(total_training_steps * WARMUP_PROPORTION)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_training_steps
)

# --- MECHANIZM WZNAWIANIA TRENINGU ---
start_epoch = 0
best_val_loss = float('inf')

if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    logger.info(f"--- Loading checkpoint from {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    logger.info(f"Successfully loaded! Resuming training from epoch {start_epoch + 1}.")

# --- GŁÓWNA PĘTLA TRENINGOWA ---
for epoch in range(start_epoch, EPOCHS):
    logger.info(f"--- Starting Epoch {epoch+1}/{EPOCHS} ---")
    
    # TRENING
    model.train() 
    train_loss, train_correct, train_total = 0, 0, 0
    
    # Zmienne do śledzenia targetu dla poszczególnych grup predykcji
    train_sum_target = 0.0

    optimizer.zero_grad() 
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        float_vectors = batch['float_vectors'].to(device)
        labels = batch['labels'].to(device).float().squeeze() 

        logits = model(input_ids, attention_mask, float_vectors).squeeze()
        
        loss = criterion(logits, labels)
        loss = loss / BATCH_ACCUMULATION 
        loss.backward()

        if ((batch_idx + 1) % BATCH_ACCUMULATION == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad() 

        train_loss += loss.item() * BATCH_ACCUMULATION 
        
        preds = (torch.sigmoid(logits) >= 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        # MASKI PREDYKCJI: Grupowanie targetów na podstawie przewidywań
        train_sum_target += labels.sum().item()

        if (batch_idx + 1) % 10 == 0:
            batch_acc = (preds == labels).sum().item() / labels.size(0)
            current_lr = scheduler.get_last_lr()[0] 
            
            avg_target = train_sum_target / train_total
            
            logger.info(f"Epoch {epoch+1} | Train Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item() * BATCH_ACCUMULATION:.4f} | Acc: {batch_acc:.2%} | Avg Target [{avg_target}]")

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    train_avg_target = train_sum_target / train_total
    
    # WALIDACJA
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_sum_target = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            float_vectors = batch['float_vectors'].to(device)
            labels = batch['labels'].to(device).float().squeeze()

            logits = model(input_ids, attention_mask, float_vectors).squeeze()
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            val_sum_target += labels.sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1} | Val Batch {batch_idx+1}/{len(val_loader)} - Validation running...")
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_avg_target = val_sum_target / val_total
    
    logger.info(f">>> End of epoch {epoch+1}")
    logger.info(f"TRAIN -> Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2%} | Avg Target {train_avg_target:.3f}")
    logger.info(f"VAL   -> Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2%} | Avg Target {val_avg_target:.3f}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        logger.info(f"*** New best model saved! (Val Loss: {best_val_loss:.4f}) ***")
    
    logger.info("-" * 50)

# --- TESTOWANIE NAJLEPSZEGO MODELU ---
logger.info("--- Training Complete. Loading best model for Testing ---")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

test_loss, test_correct, test_total = 0, 0, 0
test_sum_target = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        float_vectors = batch['float_vectors'].to(device)
        labels = batch['labels'].to(device).float().squeeze()

        logits = model(input_ids, attention_mask, float_vectors).squeeze()
        loss = criterion(logits, labels)
        
        test_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

        test_sum_target += labels.sum().item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Testing in progress... Batch {batch_idx+1}/{len(test_loader)}")

avg_test_loss = test_loss / len(test_loader)
test_accuracy = test_correct / test_total
test_avg_target = test_sum_target / test_total


logger.info(f">>> FINAL TEST RESULTS <<<")
logger.info(f"TEST -> Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.2%} | Avg Target {test_avg_target:.3f}")