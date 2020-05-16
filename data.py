import torch
import pandas as pd 
import numpy as np 
from tqdm import tqdm

tqdm.pandas()


class TransformerDataset(torch.utils.data.Dataset):
    """
    Dataset for training model
    """
    def __init__(self,
                 data_path: str,
                 max_len: int,
                 vocab_size: int):
        """
        input:
        + data_path: path to .csv file containing training data
        + max_len: max training sequence length
        + vocab_size: self-explanatory
        """
        data = pd.read_csv(data_path)

        transform_data = lambda x: np.array([int(item) for item in x.split(",")])
        self.source_sequences = data.source.progress_apply(transform_data)
        self.dest_sequences = data.dest.progress_apply(transform_data)

        self.max_len = max_len

        self.sos = torch.tensor([vocab_size]).int() # start of sentence
        self.eos = torch.tensor([vocab_size + 1]).int() # end of sentence


    def __len__(self):
        return len(self.source_sequences)

    @staticmethod
    def generate_mask(padded_seq):
        mask = padded_seq == 0
        return mask.bool()

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def __getitem__(self, index):
        # get source sequence
        padded_source_sequence = torch.zeros((self.max_len))
        source_sequence = torch.from_numpy(self.source_sequences[index]).float()
        padded_source_sequence[:len(source_sequence)] += source_sequence

        # get dest sequence
        padded_input_dest_sequence = torch.zeros((self.max_len))
        padded_target_dest_sequence = torch.zeros((self.max_len))
        dest_sequence = torch.from_numpy(self.dest_sequences[index])
        input_dest_sequence = dest_sequence[:-1].float()
        target_dest_sequence = dest_sequence[1:].float()
        padded_input_dest_sequence[:len(input_dest_sequence)] += input_dest_sequence
        padded_target_dest_sequence[:len(target_dest_sequence)] += target_dest_sequence

        # get key padding mask
        source_key_padding_mask = self.generate_mask(padded_source_sequence)
        dest_key_padding_mask = self.generate_mask(padded_input_dest_sequence)

        # get loss mask
        loss_mask = torch.bitwise_not(dest_key_padding_mask)

        return padded_source_sequence.long(), \
               padded_input_dest_sequence.long(), \
               padded_target_dest_sequence.long(), \
               source_key_padding_mask, \
               dest_key_padding_mask, \
               loss_mask
