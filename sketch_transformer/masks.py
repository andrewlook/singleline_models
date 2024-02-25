import torch


def create_padding_mask(seq):
  """
  In seq, the 5th entry in the last dimension is the padding column, which will
  be 1 if the row is padding.

  In this case, we're just inverting that field to get a padding mask. Note:
  this will not work for tokenizer-based sequences.

  :param seq: (batch_size, seq_len, 5)
  :return: (batch_size, seq_len)
  """
  return torch.abs(seq[..., -1]-1)


def create_lookahead_mask(seq_len):
  return torch.triu(torch.ones(seq_len, seq_len), diagonal=1)


def create_masks(input_seq, target_seq, device='cuda'):

    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input_seq)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(input_seq)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_lookahead_mask(target_seq.shape[1])
    dec_target_padding_mask = create_padding_mask(target_seq)

    # print(target_seq.shape, look_ahead_mask.shape, dec_target_padding_mask.shape)

    # NOTE: torch nn.MHA takes separate padding & attn masks w/ different shapes,
    #       so use that instead of combining here. TODO: check the source for how
    #       they combine.
    #
    # # TODO: WTF is combined_mask used for???
    # # TODO: can I verify this...?
    # combined_mask = torch.fmax(look_ahead_mask, dec_target_padding_mask.unsqueeze(1))

    return enc_padding_mask.to(device), dec_padding_mask.to(device), dec_target_padding_mask.to(device), look_ahead_mask.to(device)


def make_dummy_input(total_seq_len, nattn, batch_size):
  nignore = total_seq_len - nattn
  return torch.cat([
      torch.ones(batch_size, nattn, 5) * torch.tensor([0., 0., 0., 0., 0.]),
      torch.ones(batch_size, nignore, 5) * torch.tensor([0., 0., 0., 0., 1.])
  ], dim=1)