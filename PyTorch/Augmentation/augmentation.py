import torch 
import torchvision 
import torchvision.transforms as transforms 
from custom_dataset_image import CatsAndDogsDataset
from torch.utils.data import (DataLoader)
from torchvision.utils import save_image


transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = CatsAndDogsDataset(csv_file='PyTorch/archive/cats_dogs.csv', root_dir='PyTorch/archive/cats_dogs_resized', transform=transformations)

dataloader = DataLoader(dataset)

cnt = 0
for img, label in dataloader:
    cnt += 1
    save_image(img, fp='img'+str(cnt)+'.png')


