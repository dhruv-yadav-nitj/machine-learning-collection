'''
About: 
Imbalanced Datasets are those in which the number of samples of one class is much more than the other class/es. \
    Clearly it would affect the model in negative manner.
Methods: \
    1. Oversampling the minority class \
    2. Undersampling the majority class \
    3. Class Weighting (assign higher loss weights to minority class)
Example: 
Suppose you are working on a dataset with images from two classe viz. Golden_Retriver and Husky \
We have 50 samples for Golden_Retriver but only 1 samples for Husky. This is a highly biased dataset.
'''

from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import (WeightedRandomSampler, DataLoader)


# Method 2 : Class Weighting
class_count = torch.tensor([1000, 100])  # example
weights = 1.0 / class_count
loss_fn = nn.CrossEntropyLoss(weight=weights)


# Method 1 : Oversampling
def get_loader(root_dir: str, batch_size: int = 32):
    my_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])

    dataset: Tuple[torch.Tensor, int] = datasets.ImageFolder(root_dir, transform=my_transforms)
    classes: List[str] = dataset.classes

    class_weights = []

    for cls in classes:
        files = os.listdir(os.path.join(root_dir, cls))
        class_weights.append(1/len(files))  # minority class has greater class_weight

    # print(class_weights)
    # print(len(dataset), dataset[0])

    sample_weight = [0 for _ in range(len(dataset))]
    for index, (data, label) in enumerate(dataset):
        sample_weight[index] = class_weights[label]

    sampler = WeightedRandomSampler(sample_weight, len(dataset), replacement=True)

    loader = DataLoader(dataset, batch_size, sampler=sampler)
    return loader


def main():
    loader = get_loader('PyTorch/Imbalanced Dataset/dataset')
    golden_retriever = husky = 0
    for epoch in range(10):
        for data, label in loader:
            # print(label)
            golden_retriever += torch.sum(label == 0)
            '''
            if label was a numpy array -> np.sum(label==0) \
                numpy array and pytorch tensors support elementwise operations but not python default lists
            '''
            husky += torch.sum(label == 1)

    print(golden_retriever, husky)


if __name__ == '__main__':
    main()
