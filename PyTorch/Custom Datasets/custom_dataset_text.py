'''
Resource: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
Data Source: https://www.kaggle.com/datasets/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
'''


# Google Colab: Use 'kaggle.json' Authentication
'''
!pip install kaggle 
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d aladdinpersson/flickr8kimagescaptions
!unzip flickr8kimagescaptions.zip -d /content/flickr8k/
'''

'''
Resource: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
Data Source: https://www.kaggle.com/datasets/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb/download?datasetVersionNumber=1
'''

import torch
from torch.utils.data import (Dataset, DataLoader)
import os
import sys
import pandas as pd
import spacy
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from typing import List
from torch.nn.utils.rnn import pad_sequence  # used for padding -> reading description


nlp = spacy.load('en_core_web_sm')


class Vocabulary():
    def __init__(self, freq_threshold: int):
        self.count = freq_threshold  # min no. of times a word must appear corpus so that it can be added in the vocabulary
        self.itos = {
            0: 'PAD',
            1: 'SOS',
            2: 'EOS',
            3: 'UNK'
        }
        self.stoi = {
            'PAD': 0,
            'SOS': 1,
            'EOS': 2,
            'UNK': 3
        }

    @staticmethod
    def tokenizer(text) -> List[str]:
        doc = nlp(text)
        return [token.text.lower() for token in doc]

    def buildVocabulary(self, sentence_list) -> None:
        mp = defaultdict(int)
        index = 4  # index = 0,1,2,3 are already reserved for SOS, EOS, etc
        for sentence in sentence_list:
            for word in Vocabulary.tokenizer(sentence):
                mp[word] += 1
                if mp[word] == self.count:
                    self.itos[index] = word
                    self.stoi[word] = index
                    index += 1

    def numericalize(self, text):
        tokenized_text = Vocabulary.tokenizer(text)
        return [
            self.stoi[word] if word in self.stoi else self.stoi['UNK'] for word in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, img_dir, cap_dir, transform=None, freq_threshold: int = 5):
        self.transform = transform
        self.threshold = freq_threshold
        self.df = pd.read_csv(cap_dir)  # csv_file containing images_file_name, captions

        self.images = self.df['image']
        self.captions = self.df['caption']

        self.img_dir = img_dir
        self.cap_dir = cap_dir

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.buildVocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        img_file = self.images[index]
        caption = self.captions[index]

        img = Image.open(fp=os.path.join(self.img_dir, img_file)).convert(mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        # convert caption to something like numerical-vectors
        numerical_vector = [self.vocab.numericalize('SOS')]
        numerical_vector.extend(self.vocab.numericalize(caption))
        numerical_vector.append(self.vocab.numericalize('EOS'))

        return (img, torch.tensor(numerical_vector))


'''
A custom collate_fn can be used to customize collation, e.g., padding sequential data to max length of a batch. See this section on more about collate_fn.
'''


class CustomCollate():
    def __init__(self, pad_index: int):
        self.padding = pad_index  # <'PAD'>

    def __call__(self, batch):  # batch is the collection of samples (image, caption)
        imgs = [torch.unsqueeze(item[0], dim=0) for item in batch]  # (C, H, W) -> (1, C, H, W)
        imgs = torch.cat(imgs, dim=0)  # shape = (batch_size, C, H, W)

        caps = [item[1] for item in batch]
        caps = pad_sequence(caps, batch_first=False, padding_value=self.padding)  # shape = (max_seq_len, batch_size) -> default format for rnn-based models

        return (imgs, caps)


def get_loader(img_folder, annotation_file, transforms=None, batch_size=32, shuffle=True, pin_memory=2, num_workers=8):
    dataset = FlickrDataset(img_folder, annotation_file, transforms)
    pad_index = dataset.vocab.stoi['PAD']
    collate = CustomCollate(pad_index)
    loader = DataLoader(dataset, batch_size, shuffle=shuffle, sampler=None, num_workers=num_workers, collate_fn=collate, pin_memory=pin_memory)
    return (loader, dataset)


if __name__ == '__main__':
    img = '/content/flickr8k/flickr8k/images'
    cap = '/content/flickr8k/flickr8k/captions.txt'
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    loader, dataset = get_loader(img, cap, my_transforms)
    for index, (images, captions) in enumerate(loader):
        print(images.shape, captions.shape)
