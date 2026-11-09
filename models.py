import torch
from torch import nn
from torchvision import models

class SmallCNN(nn.Module):
    def __init__(self, n_classes = 15):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3,32),
                                      *block(32,64),
                                      *block(64,128),
                                      *block(128,128))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_resnet_frozen(n_classes=15):
    model = models.resnet18(weights= models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc= nn.Linear(model.fc.in_features, n_classes)
    return model

def unfreeze_layer4(model):
    for p in model.layer4.parameters():
        p.requires_grad = True
    return torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ])
