import torch
import torch.nn as nn
from transformers import DistilBertModel

# ================= CUSTOM MODEL CLASS =================
class AccidentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:,0]
        return self.out(self.dropout(pooled))

# ================= LOAD MODEL =================
model = AccidentClassifier()

model_path = "models/distilbert_multilabel_traffic.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.eval()

print("✅ MODEL LOADED SUCCESSFULLY")
