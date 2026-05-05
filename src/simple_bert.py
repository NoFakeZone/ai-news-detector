import torch
import torch.nn as nn
from transformers import AutoModel

class MultiModalBertModel(nn.Module):
    """
    bert_model_name - allegro/herbert-base-cased or dkleczek/bert-base-polish-uncased-v1
    """
    def __init__(self, bert_model_name, vector_input_size, num_classes):
        super(MultiModalBertModel, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, float_vectors):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(text_features)
        return logits