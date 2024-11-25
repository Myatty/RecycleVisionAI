# model.py
import torch
import torch.nn as nn
from torchvision import models

class RecycleClassifier(nn.Module):
    def __init__(self):
        super(RecycleClassifier, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Linear(self.features.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        x = self.features(x)
        return x

def load_model(path):
    model = RecycleClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
