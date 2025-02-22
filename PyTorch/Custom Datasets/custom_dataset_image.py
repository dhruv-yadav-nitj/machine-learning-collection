import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (Dataset, DataLoader)
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform: None):
        self.annotations = pd.read_cs(csv_file)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_path = os.join(self.root, self.annotations.iloc[index, 0])
        image = Image.open(file_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            image = self.transforms(image)

        return (image, y_label)


dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='PyTorch/archive', transform=transforms.ToTensor())

# train test splitting
train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])

train = DataLoader(train_set, shuffle=True)
test = DataLoader(test_set, shuffle=False)
