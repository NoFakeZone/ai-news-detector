import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NewsPopularityDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, tokenizer_name, max_length=512, use_features=True, max_popularity_index=1, min_popularity_index=0):
        self.texts = texts
        self.numeric_features = numeric_features
        if use_features:
            self.numeric_features[:, 20] = (self.numeric_features[:, 20] - min_popularity_index) / (max_popularity_index - min_popularity_index)
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
            'labels': label.to(torch.long)
        }
