import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        return self.embedding(input) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dp_layer = nn.Dropout(dropout)

        self.pe = torch.zeros(size=(max_seq_len, d_model))
        positions = torch.arange(0, max_seq_len, 1).unsqueeze(1)  # shape: (max_seq_len, 1)
        denominator = torch.exp(torch.arange(0, max_seq_len, 2).float() * (-math.log(10000.0)/d_model))

        self.pe[:, 0::2] = torch.sin(positions * denominator)
        self.pe[:, 1::2] = torch.cos(positions * denominator)

        self.pe = self.pe.unsqueeze(0)  # new shape : (1, max_seq_len, d_model)

        self.register_buffer('pe', self.pe)  # pe -> not considered a model params

    def forward(self, enc):
        enc += (self.pe[:, :enc.shape[1], :]).requires_grad(False)
        return self.dp_layer(enc)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # alpha & beta are model params
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.var = torch.var(x, dim=-1, keepdim=True)
        return self.alpha(x - self.mean / math.sqrt(self.var + self.eps)) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dff, d_model)
        )

    def forward(self, x):
        return self.model(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float, h: int = 8):
        super().__init__()
        self.d_model = d_model
        self.h = h  # no. of heads
        self.dp = dropout

        self.dk = d_model//h  # fraction of embedding that each head would get

        self.wq = nn.Linear(d_model, d_model)  # Wq (Query)
        self.wk = nn.Linear(d_model, d_model)  # Wk (Key)
        self.wv = nn.Linear(d_model, d_model)  # Wv (Value)

        self.wo = nn.Linear(d_model, d_model)  # Wo (Output)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None, dropout: nn.Dropout = None):
        d_k = query.shape[-1]

        # (b, h, s, dk) -> (b, h, s, s) : shape
        attention_scores = torch.matmul(query, torch.transpose(key, dim0=-2, dim1=-1))

        # masking (later?)
        '''
        masking is done here to avoid \
            reinitialisation of attention \
                for multi-head attention block
        '''
        if mask is not None:
            pass

        attention_scores = attention_scores.softmax(-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (torch.matmul(attention_scores, value), attention_scores)

    def forward(self, q, k, v, mask):
        query: torch.Tensor = self.wq(q)
        key: torch.Tensor = self.wk(k)
        value: torch.Tensor = self.wv(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1, 2)
        '''
        old_shape (32, 10, 512) -> new_shape (32, 10, 8, 64) \
            we have split each column in 8 parts \
                after transpose_shape (32, 8, 10, 64)
        '''

        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dp)
        # x: shape -> (batch, h, seq_len, dk)
        # x -> Attention(Q, K, V)

        x = torch.transpose(x, 1, 2).contiguous().view(x.shape[0], -1, self.h*self.dk)  # concatinating

        x = self.wo(x)

        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dp = dropout
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dp(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feedforward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feedforward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dp = dropout

        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tar_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tar_mask))
        x = self.residual_connection[1](x, lambda x: self.self_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tar_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tar_mask)
        return self.norm(x)


class LinearLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tar_embedding: InputEmbeddings, src_pos: PositionalEncoding, tar_pos: PositionalEncoding, projection_layer: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embedding
        self.tgt_embed = tar_embedding
        self.src_pos = src_pos
        self.tgt_pos = tar_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, Nx: int = 6, dropout: float = 0.1, dff: int = 2048):
    # creating the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # creating positional embeddings
    src_pos = PositionalEncoding(d_model, dropout, src_seq_len)
    tgt_pos = PositionalEncoding(d_model, dropout, src_seq_len)

    # encoder
    encoder_blocks = []
    for _ in range(Nx):
        attention_block = MultiHeadAttention(d_model, dropout, h)
        ffnn = FeedForward(d_model, dff, dropout)
        encoder_block = EncoderBlock(attention_block, ffnn, dropout)
        encoder_blocks.append(encoder_block)

    # decoder
    decoder_blocks = []
    for _ in range(Nx):
        attention_block = MultiHeadAttention(d_model, dropout, h)
        cross_attention = MultiHeadAttention(d_model, dropout, h)
        ffnn = FeedForward(d_model, dff, dropout)
        decoder_block = DecoderBlock(attention_block, cross_attention, ffnn, dropout)
        decoder_blocks.append(decoder_block)

    # encoder & decoder
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Encoder(nn.ModuleList(decoder_block))

    projection_layer = LinearLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:  # weight initialisation (biases having dim=1 are set to zeros)
            nn.init.xavier_uniform(p)

    return transformer
