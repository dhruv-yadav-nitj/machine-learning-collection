from datasets import load_dataset
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from typing import Dict
from torch.utils.data import random_split, DataLoader
from dataset import BilingualDataset
from model import build_transformer
import torch.optim as optim
from config import get_config, get_weights_file_path
import tqdm

def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        # returns an iterator rather than a value -> efficient for larger datasets


def get_or_build_tokenizer(config: Dict, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))  # return path in form of object
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(model=WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(iterator=get_all_sentence(dataset, lang), trainer=trainer)
        tokenizer.save(path=str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(path=str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    ds_raw = load_dataset('opus_books', f'{config['src_lang']}-{config['tgt_lang']}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])

    ds_size = len(ds_raw)
    train_size = int(0.9 * ds_size)
    val_size = ds_size - train_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])  # tokenized dataset

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])

    train_dataloader = DataLoader(train_ds, config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, 1)

    return (train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt)


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # reset states
    initial_epochs = global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)
        initial_epochs = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(['PAD']), label_smoothing=0.1).to(device)

    for epoch in range(initial_epochs, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch: 02d}')

