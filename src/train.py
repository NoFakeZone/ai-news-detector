import os
import glob
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

from dataset import NewsPopularityDataset
from feature_bert import MultiModalBertModel

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    dataset_dir = "dataset"
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    logger.info(f"Found classes: {class_to_idx}")
    
    files = glob.glob(os.path.join(dataset_dir, "*", "*.json"))
    logger.info(f"Found {len(files)} JSON files. Loading data...")
    
    data = []
    providers = []
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            try:
                item = json.load(file)
            except json.JSONDecodeError:
                continue
            
            provider = item.get("Provider", "unknown")
            text = item.get("Wygenerowany tytu\u0142", "") + " " + item.get("Wygenerowany tekst", "")
            label = class_to_idx[os.path.basename(os.path.dirname(f))]
            
            data.append({"text": text, "provider": provider, "label": label})
            providers.append(provider)
            
    logger.info(f"Loaded {len(data)} valid items. Splitting data...")
            
    # Split first, calculate on train data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}")
    
    train_providers = [item["provider"] for item in train_data]
    provider_counts = Counter(train_providers)
    max_count = max(provider_counts.values()) if provider_counts else 1
    
    def process_data(dataset_split):
        texts = [item["text"] for item in dataset_split]
        numeric_features = [[provider_counts.get(item["provider"], 0) / max_count] for item in dataset_split]
        labels = [item["label"] for item in dataset_split]
        return NewsPopularityDataset(texts, numeric_features, labels, "allegro/herbert-base-cased")

    train_dataset = process_data(train_data)
    val_dataset = process_data(val_data)
    
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    model = MultiModalBertModel("allegro/herbert-base-cased", vector_input_size=1, num_classes=len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    logger.info(f"Starting training on {device} with {len(train_dataset)} samples...")
    
    epochs = 1
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            float_vectors = batch['float_vectors'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, float_vectors)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                
        logger.info(f"Epoch {epoch} completed.")
        
if __name__ == '__main__':
    main()
