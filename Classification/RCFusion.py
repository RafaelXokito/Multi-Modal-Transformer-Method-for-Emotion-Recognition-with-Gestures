import torch
import torch.nn as nn
import torchvision.models as models

class RCFusion(nn.Module):
    def init(self, num_classes):
        super().init()

        # RGB Network
        self.rgb_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.rgb_feature_extractor = nn.Sequential(list(self.rgb_feature_extractor.children())[:-1])

        # Depth Network
        self.depth_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.depth_feature_extractor = nn.Sequential(list(self.depth_feature_extractor.children())[:-1])
        self.depth_feature_extractor[0] = nn.Conv2d(1, 64, kernel_size=7)

        # Fusion
        self.gru = nn.GRU(input_size=1024, hidden_size=128, num_layers=2, dropout=0.4)

        # Classification
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.rgb_feature_extractor(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.depth_featureextractor(x2)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = x.unsqueeze(0)
        , hidden = self.gru(x)
        x = self.fc(hidden[-1])
        x = self.softmax(x)
        return x