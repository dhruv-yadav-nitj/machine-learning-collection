import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.src_tokenizer: Tokenizer = tokenizer_src
        self.tgt_tokenizdr: Tokenizer = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id(['SOS'])], dtype=torch.int64)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id(['EOS'])], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id(['PAD'])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        target_pair = self.ds[index]
        src_text = target_pair['translation'][self.src_lang]
        tgt_text = target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.src_tokenizer.encode(src_text).ids  # array
        dec_input_tokens = self.tgt_tokenizdr.encode(tgt_text).ids

        required_padding_tokens_enc = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
        required_padding_tokens_dec = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS only (in case of Decoder)

        if required_padding_tokens_enc < 0 or required_padding_tokens_dec < 0:
            raise ValueError('Sentence is too long!')

        enc_input = torch.cat(tensors=[
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * required_padding_tokens_enc, dtype=torch.int64)
        ])

        dec_input = torch.cat(tensors=[
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * required_padding_tokens_dec, dtype=torch.int64)
        ])

        # target o/p (the ground truth) for training the model -> thats why there is no ['SOS'] token
        # the o/p we are expecting from the decoder in the target language
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * required_padding_tokens_dec, dtype=torch.int64)
        ])

        return {
            'encoder_input': enc_input,
            'decoder_input': dec_input,
            'encoder_mask': (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & BilingualDataset.causal_mask(dec_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).dtype(torch.int)  # everything below the diagonal becomes 0
        return mask == 0  # but we want everythin below the diagnoal to be 1 and above to be 0
