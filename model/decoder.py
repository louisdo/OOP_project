import torch
from .positional_encoding import PositionalEncoding 


class Decoder(torch.nn.Module):
    """
    Transformer Decoder. 
    For more information on the architecture, please refer to
    the original paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dropout: float,
                 vocab_size: int,
                 max_len: int):

        """
        input: 
        + d_model: dimensionality of Transformer
        + nhead: number of heads for multi-head attention
        + num_layers: number of layers for decoder
        + dropout: dropout rate
        + vocab_size: self-explanatory
        + max_len: max length of training sequence
        """
        super(Decoder, self).__init__()

        self.posenc = PositionalEncoding(d_model = d_model,
                                         dropout = dropout,
                                         max_len = max_len)

        decoder_layer = torch.nn.modules.TransformerDecoderLayer(d_model = d_model,
                                                                 nhead = nhead,
                                                                 dropout = dropout)
        self.decoder = torch.nn.modules.TransformerDecoder(decoder_layer = decoder_layer,
                                                           num_layers = num_layers)

        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(in_features = d_model,
                                      out_features = vocab_size,
                                      bias = True)

        
    def forward(self, 
                embedded_target: torch.tensor, 
                memory: torch.tensor, 
                tgt_mask = None,
                tgt_key_padding_mask: torch.ByteTensor = None, 
                memory_key_padding_mask: torch.ByteTensor = None):
        
        """
        input:
        + embedded_target: torch.tensor of shape [L, B, d_model] with 'L' be the max_len,
        'B' be the batchsize and 'd_model' is d_model; the embedded vectors of the source 
        sequence.
        + memory: torch.tensor of shape [L, B, d_model] with 'L' be the max_len,
        'B' be the batchsize and 'd_model' is d_model; the output of the Transformer Encoder
        + tgt_key_padding_mask: torch.ByteTensor of shape [B, L] with 'B' be the batchsize,
        'L' be the max_len; padding mask to ignore the paddings, this mask will be used for the 'embedded_target'.
        + memory_key_padding_mask: torch.ByteTensor of shape [B, L] with 'B' be the batchsize,
        'L' be the max_len; padding mask to ignore the paddings, this mask will be used for the 'memory'

        output:
        + decoder output
        """

        embedded_target = self.posenc(embedded_target)
        output = self.relu(self.decoder(embedded_target, 
                                        memory,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask = tgt_key_padding_mask,
                                        memory_key_padding_mask = memory_key_padding_mask))
        output = self.linear(output)
        
        return output