import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config

class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.bert_model_name)
        
        # # Freeze the BERT model
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)  # Output size changed to num_classes for multi-class classification
        self.layernorm1 = nn.LayerNorm(512)
        self.layernorm2 = nn.LayerNorm(256)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.layernorm2(x)
        x = self.dropout(x)

        logits = self.fc3(x)  # No need for softmax activation here for multi-class classification
        return logits