import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from feature_bert import MultiModalBertModel 

# --- KONFIGURACJA I PARAMETRY ---
BERT_MODEL_NAME = "allegro/herbert-base-cased" #or dkleczek/bert-base-polish-uncased-v1
VECTOR_SIZE = 15  
NUM_CLASSES = 2 
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- INICJALIZACJA MODELU ---
model = MultiModalBertModel(
    bert_model_name=BERT_MODEL_NAME, 
    vector_input_size=VECTOR_SIZE, 
    num_classes=NUM_CLASSES
)
model.to(device)

# --- OPTYMALIZATOR I FUNKCJA STRATY ---
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# --- GŁÓWNA PĘTLA TRENINGOWA ---
for epoch in range(EPOCHS):
    model.train() 
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        float_vectors = batch['float_vectors'].to(device)
        labels = batch['labels'].to(device).long() 

        optimizer.zero_grad() # Czyścimy stare gradienty
        logits = model(input_ids, attention_mask, float_vectors) # Forward pass
        loss = criterion(logits, labels) # Obliczanie straty
        loss.backward() # Backward pass (obliczanie gradientów)
        optimizer.step() # Aktualizacja wag

        total_loss += loss.item()
        # Obliczanie dokładności (accuracy)
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Podsumowanie epoki
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions.double() / total_samples
    print(f"\n>>> KONIEC EPOKI {epoch+1} | Średnia Strata: {avg_loss:.4f} | Accuracy: {accuracy:.4f}\n")

# --- ZAPISYWANIE MODELU ---
torch.save(model.state_dict(), "trained_bert_stylistic_model.pt")
print("Model został zapisany!")