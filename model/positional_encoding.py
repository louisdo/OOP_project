import torch
import math


class PositionalEncoding(torch.nn.Module):
    """
    Transformer Positional Encoding
    For more information on this operation, please refer to the paper:
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float, 
                 max_len: int):
        """
        input: 
        + d_model: dimensionality of Transformer
        + dropout: dropout rate
        + max_len: max length of training sequence
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input:
        + x: torch.tensor of shape [L, B, d_model], with 'L' be the 
        max training sequence length, 'B' be the batchsize, 'd_model' be d_model;
        the embedded sequence you would like to positional encode

        output:
        + positional encoded embedded sequence
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)