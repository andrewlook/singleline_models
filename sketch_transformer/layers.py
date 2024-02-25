import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):

    def __init__(self,
                 enc_hidden_size: int, # d_model
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 dropout_prob=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(enc_hidden_size, num_heads, dropout=dropout_prob, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(enc_hidden_size, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, enc_hidden_size),
            nn.Dropout(dropout_prob),
        )
        self.ln1 = nn.LayerNorm([max_len, enc_hidden_size]) # -1 instead of batch size?
        self.ln2 = nn.LayerNorm([max_len, enc_hidden_size])
        # self.ln1 = nn.LayerNorm([hp.max_seq_length+2, enc_hidden_size]) # -1 instead of batch size?
        # self.ln2 = nn.LayerNorm([hp.max_seq_length+2, enc_hidden_size])

    def forward(self, x: torch.Tensor, mask=None):
        """
        x will have shape `[seq_len, batch_size, 5]`
        """

        attn_output, _ = self.mha(x, x, x, mask)
        out1 = self.ln1(x + attn_output)

        ffn_output = self.ffn(out1)
        out2 = self.ln2(out1 + ffn_output)
        
        return out2
    

class DecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 dropout_prob=0.0):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )
        self.ln1 = nn.LayerNorm([max_seq_len, d_model]) # -1 instead of batch size?
        self.ln2 = nn.LayerNorm([max_seq_len, d_model])
        self.ln3 = nn.LayerNorm([max_seq_len, d_model])

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, padding_mask, dec_target_padding_mask, look_ahead_mask):
        """
        x will have shape `[batch_size, target_seq_len, d_model]`
        enc_output will have shape `[batch_size, input_seq_len, d_model]`
        """

        attn1, attn1_weights = self.mha1(x, x, x, key_padding_mask=dec_target_padding_mask, attn_mask=look_ahead_mask)
        out1 = self.ln1(x + attn1)

        attn2, attn2_weights = self.mha2(enc_output, enc_output, out1, padding_mask)
        out2 = self.ln2(out1 + attn2)

        ffn_output = self.ffn(out2)
        out3 = self.ln3(ffn_output)
        
        return out3, attn1_weights, attn2_weights
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 252, batch_size: int =100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        pe = torch.zeros(batch_size, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch, seq_len, embedding_dim]``
        """
        x_pe = self.pe[:, :x.size(1)]
        # print(f"x.shape: {x.shape}, x_pe.shape: {x_pe.shape}")
        x = x + x_pe
        return self.dropout(x)
    


class SelfAttn(nn.Module):
    """
    Compute a single embedding based on a whole sequence of embedding outputs from
    multi-head attention layers, as described in End-to-End Memory Networks:
    https://arxiv.org/abs/1503.08895
    """

    def __init__(self, d_model, d_lowerdim):
        super().__init__()
        self.embedding_layer = nn.Linear(d_model, d_lowerdim)
        self.W = nn.Parameter(torch.randn(d_model, d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.V = nn.Parameter(torch.rand(d_model, 1))
    
    def forward(self, x):
        """
        u_i = tanh(xW + b)
        a_i = softmax(u_i * V)
        o = sum(a * x)

        :param x: input tensor (batch_size, seq_len, d_model)
        :return:  output tensor (batch_size, d_lowerdim)
        """
        u_i = F.tanh((x @ self.W) + self.b) # (batch_size, seq_len, d_model)
        a_i = F.softmax(u_i @ self.V, dim=1) # (batch_size, seq_len, 1)
        o = torch.sum(x * a_i, dim=1) # (batch_size, seq_len, d_model)
        o = self.embedding_layer(o)
        return o, a_i
  

class DenseExpander(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len):
        super().__init__()
        self.project_layer = nn.Linear(in_dim, out_dim)
        self.expand_layer = nn.Linear(1, seq_len)

    def forward(self, x):
        x = F.relu(self.project_layer(x))
        x = x.unsqueeze(2) # (batch_size, out_dim, 1)
        x = self.expand_layer(x) # (batch_size, out_dim, seq_len)
        x = x.transpose(1, 2) # (batch_size, seq_len, out_dim)
        return x