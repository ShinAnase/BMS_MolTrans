import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

#Resnet(画像認識転移学習モデルの部分)
class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x):
        bs = x.size(0)
        features = self.cnn.forward_features(x)
        features = features.permute(0, 2, 3, 1)
        return features