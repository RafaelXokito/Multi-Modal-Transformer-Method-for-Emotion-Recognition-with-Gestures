from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os

class CeramicNet(Dataset):
    def __init__(self, rgb_folder, depth_folder, transform=None):
        self.num_classes = len(os.listdir(rgb_folder))
        self.rgb_dataset = datasets.ImageFolder(root=rgb_folder, transform=transform)
        depth_transform = transforms.Compose([transforms.Grayscale(), transform])
        self.depth_dataset = datasets.ImageFolder(root=depth_folder, transform=depth_transform)

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, index):
        rgb, label = self.rgbdataset[index]
        depth,  = self.depth_dataset[index]
        return rgb, depth, label