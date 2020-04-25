import torch


class WordEmb(torch.nn.Module):
    """
    Word2Vec module
    """
    def __init__(self, vocab_size, d_model):
        """
        input:
        + vocab_size: the vocab size
        + d_model: the dimension of the desired embedding
        """
        super(WordEmb, self).__init__()

        self.wordemb = torch.nn.Embedding(vocab_size, d_model)

    def forward(self, sequence: torch.tensor):
        """
        input:
        + sequence: torch.tensor of shape [L, B] with 'L' be the max training sequence length,
        'B' be the batchsize; the encoded source sequence you need to embed

        output:
        + embedding vector of input sequence
        """
        return self.wordemb(sequence)
