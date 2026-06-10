import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NewsPopularityDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, tokenizer_name, max_length=512, use_features=True, feature_means=None, feature_stds=None):
        self.texts = texts
        # Tworzymy kopię (clone) i rzutujemy na float
        self.numeric_features = numeric_features.clone().float() 
        
        if use_features and feature_means is not None and feature_stds is not None:
            # Zabezpieczenie przed dzieleniem przez zero (gdy cecha ma stałą wartość, std = 0)
            stds = feature_stds.clone()
            stds[stds == 0] = 1e-8 
            
            # Wektoryzowana normalizacja Standard Scaling: (x - mean) / std
            self.numeric_features = (self.numeric_features - feature_means) / stds
            
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        features = self.numeric_features[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'float_vectors': features.to(torch.float),
            'labels': label.clone().detach().to(torch.long)
        }