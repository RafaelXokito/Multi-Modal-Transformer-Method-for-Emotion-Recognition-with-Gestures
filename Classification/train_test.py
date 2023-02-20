import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT

# Load the pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Remove the last layer of the ResNet model
resnet_layers = list(resnet.children())[:-1]
resnet = nn.Sequential(*resnet_layers)

# Create a small ViT model with 8 attention heads
vit = ViT(
    image_size=224,
    patch_size=32,
    num_classes=1000,
    dim=768,
    depth=6,
    heads=8,
    mlp_dim=3072,
    dropout=0.1,
    emb_dropout=0.1
)

# Wrap the ResNet model with the ViT model
class ResNetViT(nn.Module):
    def __init__(self, resnet, vit):
        super().__init__()
        self.resnet = resnet
        self.vit = vit

    def forward(self, x):
        x = self.resnet(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.vit(x)
        return x

# Instantiate the ResNetViT model
model = ResNetViT(resnet, vit)

# Save the model to a file
torch.save(model.state_dict(), 'resnet_vit.pt')
