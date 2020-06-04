import torch
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from lib import Utils

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



    def get_training_instance(self, index):
        # get padded source sequence
        source_sequence = torch.from_numpy(self.source_sequences[index]).float()
        padded_source_sequence = Utils.get_padded_sequence(source_sequence, self.max_len)

        # get padded dest sequence
        dest_sequence = torch.from_numpy(self.dest_sequences[index])
        input_dest_sequence = dest_sequence[:-1].float()
        target_dest_sequence = dest_sequence[1:].float()

        padded_input_dest_sequence = Utils.get_padded_sequence(input_dest_sequence, self.max_len)
        padded_target_dest_sequence = Utils.get_padded_sequence(target_dest_sequence,self.max_len)

        # get key padding mask
        source_key_padding_mask = Utils.generate_mask(padded_source_sequence)
        dest_key_padding_mask = Utils.generate_mask(padded_input_dest_sequence)

        # get loss mask
        loss_mask = torch.bitwise_not(dest_key_padding_mask)

        return padded_source_sequence.long(), \
               padded_input_dest_sequence.long(), \
               padded_target_dest_sequence.long(), \
               source_key_padding_mask, \
               dest_key_padding_mask, \
               loss_mask


    def __getitem__(self, index):
        return self.get_training_instance(index)