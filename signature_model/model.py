import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SignatureEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=256):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.embedding_dim = 2048
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, p=2, dim=1)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()
