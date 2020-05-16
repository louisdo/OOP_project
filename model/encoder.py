import torch
from .positional_encoding import PositionalEncoding  


class Encoder(torch.nn.Module):
    """
    Transformer Encoder.
    For more information on the architecture, please refer to
    the original paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 dropout: float, 
                 max_len: int):
        """
        input: 
        + d_model: dimensionality of Transformer
        + nhead: number of heads for multi-head attention
        + num_layers: number of layers for encoder
        + dropout: dropout rate
        + max_len: max length of training sequence
        """
        
        super(Encoder, self).__init__()

        self.posenc = PositionalEncoding(d_model = d_model,
                                         dropout = dropout,
                                         max_len = max_len)

        encoder_layer = torch.nn.modules.TransformerEncoderLayer(d_model = d_model,
                                                                 nhead = nhead,
                                                                 dropout = dropout)
        self.encoder = torch.nn.modules.TransformerEncoder(encoder_layer = encoder_layer,
                                                           num_layers = num_layers)

    def forward(self, 
                embedded_source: torch.tensor,
                src_key_padding_mask = None):
        """
        input:
        + embedded_source: torch.tensor of shape [L, B, d_model], with 'L' be the 
        max training sequence length, 'B' be the batchsize, 'd_model' be d_model;
        the embedded vectors of the source sentence
        + src_key_padding_mask: torch.ByteTensor of shape [B, L] with 'B' be the batchsize,
        'L' be the max_len; padding mask to ignore the paddings, this mask will be used for the 'embedded_source'

        output:
        + encoder output
        """

        embedded_source = self.posenc(embedded_source)
        output = self.encoder(embedded_source,
                              src_key_padding_mask = src_key_padding_mask)

        return output