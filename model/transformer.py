import torch
from .encoder import Encoder
from .decoder import Decoder 
from .wordemb import WordEmb 

class Transformer(torch.nn.Module):
    """
    Transformer.
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
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model = d_model,
                               nhead = nhead,
                               num_layers = num_layers,
                               dropout = dropout,
                               max_len = max_len)
        
        self.decoder = Decoder(d_model = d_model,
                               nhead = nhead,
                               num_layers = num_layers,
                               dropout = dropout,
                               vocab_size = vocab_size,
                               max_len = max_len)
        
        # in this problem, the source and the dest sequence comes from the same language,
        # so 1 word embedder is sufficient
        self.wordemb = WordEmb(vocab_size = vocab_size,
                               d_model = d_model)

    
    def forward(self, 
                source, 
                target,
                src_key_padding_mask = None,
                tgt_mask = None,
                tgt_key_padding_mask = None):
        """
        input:
        + source: torch.tensor of shape [L, B] with 'L' be the max training sequence length,
        'B' be the batchsize; the encoded source sequence
        + target:torch.tensor of shape [L, B] with 'L' be the max training sequence length,
        'B' be the batchsize; the encoded target sequence
        + src_key_padding_mask: torch.ByteTensor of shape [B, L] with 'B' be the batchsize,
        'L' be the max_len; padding mask to ignore the paddings for the source sequence.
        + tgt_key_padding_mask: torch.ByteTensor of shape [B, L] with 'B' be the batchsize,
        'L' be the max_len; padding mask to ignore the paddings for the target sequence.

        output:
        + model output of shape [B, L, V] with 'B' be the batchsize, 'L' the the training sequence length,
        'V' be the vocab size
        """
        
        embedded_source = self.wordemb(source)
        embedded_target = self.wordemb(target)

        memory = self.encoder(embedded_source,
                              src_key_padding_mask = src_key_padding_mask)

        output = self.decoder(embedded_target, 
                              memory,
                              tgt_mask = tgt_mask,
                              tgt_key_padding_mask = tgt_key_padding_mask,
                              memory_key_padding_mask = src_key_padding_mask)

        return output.transpose(0, 1)