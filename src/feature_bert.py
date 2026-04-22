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
        
        self.mlp = nn.Sequential(
            nn.Linear(vector_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        combined_features_size = bert_hidden_size + 32
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, float_vectors):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :]
        vec_features = self.mlp(float_vectors)
        combined = torch.cat((text_features, vec_features), dim=1)
        logits = self.classifier(combined)
        return logits