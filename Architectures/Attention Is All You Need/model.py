import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model//heads

        assert self.d_k * self.h == self.d_model, 'd_model must be divisible by h'

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, value, key, query, mask):
        n = query.shape[0]  # no. of training examples
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # value_len = key_len = query_len = seq_len (in encoder only) (things are different in decoder)

        value = self.w_v(value)  # shape -> (n, value_len, d_model)
        key = self.w_k(key)
        query = self.w_q(query)

        # dividing the K, V, Q for each head
        value: torch.Tensor = value.reshape(n, value_len, self.h, self.d_k)
        key: torch.Tensor = key.reshape(n, key_len, self.h, self.d_k)
        query: torch.Tensor = query.reshape(n, query_len, self.h, self.d_k)

        # change the dimensions for proper mat mul
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        # shape : (n, seq_len, h, d_k) -> (n, h, seq_len, d_k)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # shape: (n, h, query_len, key_len)

        # masking -> please relate this code block with the make_src_mask method in Transformer
        '''
        Purpose: to ignore the contribution of PAD in attention calculation! how?: 1e-20 is so small that it gets ignored when we do softmax
        '''
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, value=float(-1e9))  # sets value where condition is True

        attention_scores = torch.softmax(attention_scores / math.sqrt(self.d_k), dim=-1)

        out = torch.matmul(attention_scores, value)  # shape: (n, h, query_len, d_k)

        out = out.permute(0, 2, 1, 3).reshape(n, query_len, self.h * self.d_k)  # new shape: (n, seq_len, d_model)

        out = self.w_o(out)  # MultiHead(Q, K, V)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, dp: float = 0.1, d_model: int = 512, d_ff: int = 2048, heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff  # inner_layer dimension of feedforward network
        self.dp = dp
        self.h = heads

        self.attention = SelfAttention(d_model, heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(p=dp),  # uncertain whether I should keep dropout here or not? please confirm
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dp)
        )

        self.dropout = nn.Dropout(dp)  # dropout layer

    def forward(self, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor, mask):
        # self attention
        mha = self.attention(v, k, q, mask)

        x = q

        # add & norm (residual connection, sublayer input)
        x = self.dropout(self.norm1(x + mha))  # first part

        x = self.dropout(self.norm2(x + self.feedforward(x)))  # second part
        return x  # final encoder output


class Encoder(nn.Module):
    def __init__(self,
                 device,
                 src_vocab_size: int,  # vocab size of the source language
                 max_length: int,  # max length of input sequences (needed for positional encoding)
                 d_model: int = 512,
                 nx: int = 6,
                 heads: int = 8,
                 dff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.word_embedding = nn.Embedding(src_vocab_size, d_model)  # (size_of_the_dictionary_of_embedding, embed_size)
        self.positional_embedding = nn.Embedding(max_length, d_model)

        self.layers = nn.ModuleList([
            EncoderBlock(dropout, d_model, dff, heads) for _ in range(nx)
        ])

        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, x: torch.Tensor, mask):
        n, seq_len = x.shape  # batch of sentences
        positions = torch.arange(0, seq_len).expand(n, seq_len).to(self.device)
        x = self.dropout(
            self.word_embedding(x) + self.positional_embedding(positions)
        )

        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, device, d_model: int = 512, d_ff: int = 2048, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dp = dropout
        self.h = heads
        self.d_ff = d_ff
        self.device = device

        self.attention = SelfAttention(d_model, heads)

        self.common_with_encoder = EncoderBlock(dropout, d_model, d_ff, heads)
        # the top part in decoder block is same as encoder, so we can use the same code here as well

        self.dropout = nn.Dropout(p=dropout)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, value, key, src_mask, tgt_mask):
        '''
        x: decoder i/p
        value, key: coming from encoder
        '''

        mmha = self.attention(x, x, x, tgt_mask)  # masked mha
        x = self.dropout(self.norm(x + mmha))
        out = self.common_with_encoder(value, key, x, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(self,
                 device,
                 tgt_vocab_size,
                 max_length,
                 d_model: int = 512,
                 heads: int = 8,
                 nx: int = 6,
                 dff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_length, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(device, d_model, dff, heads, dropout) for _ in range(nx)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        n, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(n, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 device,
                 src_vocab_size,
                 tgt_vocab_size,
                 src_pad_idx,
                 tgt_pad_idx,
                 d_model: int = 512,
                 nx: int = 6,
                 d_ff: int = 2048,
                 heads: int = 8,
                 dropout: float = 0.1,
                 max_length: int = 100):
        super().__init__()
        self.encoder = Encoder(device, src_vocab_size, max_length, d_model, nx, heads, d_ff, dropout)
        self.decoder = Decoder(device, tgt_vocab_size, max_length, d_model, heads, nx, d_ff, dropout)

        self.src_pad_index = src_pad_idx
        self.tgt_pad_index = tgt_pad_idx

        self.device = device

    '''
    Ex:
    src = [3,5,4,0,0] ; assume 0 is the padding index
    mask -> [True, True, True, False, False]  # so we get False at those indices where we had PAD.
    now refer to the masking in forward function of SelfAttention.
    '''

    def make_src_mask(self, src):
        mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2).to(self.device)  # shape: N, 1, 1, src_len
        return mask

    # need causal mask here -> prevent current token from seeing future tokens
    def make_tgt_mask(self, tgt):
        n, tgt_len = tgt.shape
        temp_mat = torch.ones((tgt_len, tgt_len))
        mask = torch.tril(temp_mat, diagonal=0).expand(n, 1, tgt_len, tgt_len).to(self.device)
        return mask  # shape (n, 1, tgt_len, tgt_len)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, src_mask, tgt_mask)

        return dec


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0],  # src_sentence 1
        [1, 8, 7, 3, 4, 5, 6, 7, 2]  # src_sentence 2
    ]).to(device)

    y = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0],  # tgt_sentence 1
        [1, 5, 6, 2, 4, 7, 6, 2]  # tgt_sentence 2
    ]).to(device)

    src_pad_index = tgt_pad_index = 0
    src_vocab_size = 10
    tgt_vocab_size = 10

    model = Transformer(device, src_vocab_size, tgt_vocab_size, src_pad_index, tgt_pad_index).to(device)

    out = model(x, y[:, :-1])  # during training, the decoder must not see the last token of tgt as input

    print(out.shape)  # req : (batch_size, tgt_seq_len-1, tgt_vocab_size) here (2, 7, 10)
