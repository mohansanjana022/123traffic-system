import torch
from transformers import DistilBertModel

def load_model():

    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    model.load_state_dict(
        torch.load("models/distilbert_multilabel_traffic.pth",
        map_location=torch.device("cpu"))
    )

    model.eval()

    return model
