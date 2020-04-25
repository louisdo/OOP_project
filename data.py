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

    def __getitem__(self, index):
        # get source sequence
        padded_source_sequence = torch.zeros((self.max_len))
        source_sequence = torch.from_numpy(self.source_sequences[index])
        source_sequence = torch.cat([self.sos,
                                     source_sequence,
                                     self.eos]).float()
        padded_source_sequence[:len(source_sequence)] += source_sequence

        # get dest sequence
        padded_input_dest_sequence = torch.zeros((self.max_len))
        padded_target_dest_sequence = torch.zeros((self.max_len))
        dest_sequence = torch.from_numpy(self.dest_sequences[index])
        input_dest_sequence = torch.cat([self.sos,
                                         dest_sequence]).float()
        target_dest_sequence = torch.cat([dest_sequence,
                                          self.eos]).float()
        padded_input_dest_sequence[:len(input_dest_sequence)] += input_dest_sequence
        padded_target_dest_sequence[:len(target_dest_sequence)] += target_dest_sequence

        return padded_source_sequence.long(), \
               padded_input_dest_sequence.long(), \
               padded_target_dest_sequence.long()
