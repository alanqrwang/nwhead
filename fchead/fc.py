import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, featurizer, in_dim, num_classes):
        super(FCNet, self).__init__()
        self.featurizer = featurizer
        self.classifier = FCHead(in_dim, out_dim=num_classes)

    def extract_feat(self, qimg):
        return self.featurizer(qimg)

    def forward(self, inputs):
        features = self.featurizer(inputs)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

class FCHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCHead, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        return x