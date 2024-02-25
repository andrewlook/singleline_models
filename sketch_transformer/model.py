import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import (DecoderLayer, DenseExpander, EncoderLayer,
                     PositionalEncoding, SelfAttn)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Linear(5, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate, max_len=max_seq_len)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, max_seq_len=max_seq_len, dropout_prob=dropout_rate) for _ in range(num_layers)])

    def forward(self, x, mask):
        seq_len = x.shape[1]
        x = self.embedding(x) # (batch, seq, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model))
        # add positional embedding, and apply dropout
        x = self.pos_encoding(x)

        for i, layer in enumerate(self.enc_layers):
            x = layer(x, mask)
        return x # (batch, seq, d_model)
  

class Decoder(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                    max_seq_len=1000,
                    dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Linear(5, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate, max_len=max_seq_len)

        self.dec_layers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, dropout_prob=dropout_rate)
                            for _ in range(num_layers)])

    def forward(self, x, enc_output, padding_mask, dec_target_padding_mask, look_ahead_mask):
        seq_len = x.shape[1]
        x = self.embedding(x) # (batch, seq, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model))
        # add positional embedding, and apply dropout
        x = self.pos_encoding(x)

        attention_weights = {}
        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, padding_mask, dec_target_padding_mask, look_ahead_mask)

        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
    

class Model(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.encoder = Encoder(
            num_layers=hp.n_layer,
            d_model=hp.d_model,
            num_heads=hp.n_head,
            d_ff=hp.d_ff,
            maximum_position_encoding=hp.max_seq_length+2,
            dropout_rate=hp.dropout_rate)
        self.bottleneck_layer = SelfAttn(d_model=hp.d_model, d_lowerdim=hp.d_lowerdim)
        self.expand_layer = DenseExpander(in_dim=hp.d_lowerdim, out_dim=hp.d_model, seq_len=hp.max_seq_length+2)
        self.decoder = Decoder(num_layers=hp.n_layer,
            d_model=hp.d_model,
            num_heads=hp.n_head,
            d_ff=hp.d_ff,
            maximum_position_encoding=hp.max_seq_length+2,
            dropout_rate=hp.dropout_rate)
        self.output_layer = nn.Linear(hp.d_model, 5)
    
    def encode(self, x, mask):
        enc_output = self.encoder(x, mask)
        lowerdim_output, _ = self.bottleneck_layer(enc_output)
        return lowerdim_output, enc_output
    
    def decode(self, embedding, target, dec_padding_mask, dec_target_padding_mask, look_ahead_mask):
        """Generate logits"""
        padding_mask = torch.zeros_like(dec_padding_mask) if self.hp.blind_decoder_mask else dec_padding_mask
        pre_decoder = self.expand_layer(embedding)
        dec_output, attention_weights = self.decoder(target, pre_decoder, padding_mask, dec_target_padding_mask, look_ahead_mask)
        final_output = self.output_layer(dec_output)
        return final_output, attention_weights

    def forward(self, input_seq, target_seq, enc_padding_mask, dec_padding_mask, dec_target_padding_mask, look_ahead_mask):
        lowerdim_output, enc_output = self.encode(input_seq, enc_padding_mask)
        final_output, attention_weights = self.decode(lowerdim_output, target_seq, dec_padding_mask, dec_target_padding_mask, look_ahead_mask)
        return final_output, lowerdim_output #, enc_output, attention_weights

    # def encode_from_seq(self, inp_seq):
    #     """same as encode but compute mask inside. Useful for test"""
    #     dtype = tf.float32 if self.dataset.hps['use_continuous_data'] else tf.int64
    ##
    ## i think that last element of the tensor below is to account for train-time
    ## shifting off-by for for input vs target...
    ##
    #     encoder_input = tf.cast(np.array(inp_seq) + np.zeros((1, 1)), dtype)  # why?
    #     enc_padding_mask = builders.utils.create_padding_mask(encoder_input)
    #     res = self.encode(encoder_input, enc_padding_mask, training=False)
    #     return res
  

class ReconstructionLoss(nn.Module):
   
    def forward(pred, real):
        pred_locations = pred[:, :, :2]
        # pred_metadata = pred[:, :, 2:]
        tgt_locations = real[:, :, :2]
        # tgt_metadata = real[:, :, 2:]
        location_loss = F.mse_loss(pred_locations, tgt_locations) #, reduction='none')
        return location_loss
        
        # metadata_loss = F.cross_entropy(F.softmax(pred_metadata, dim=-1), F.softmax(tgt_metadata, dim=-1)) #, reduction='none')
        # return location_loss, metadata_loss
      