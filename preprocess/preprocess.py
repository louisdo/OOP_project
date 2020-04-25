import json
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import errno
import sys
sys.path.append(".")
from lib.folders import maybe_create_folder

tqdm.pandas()


class Preprocessing:
    """
    Data preprocessing utility
    """
    def __init__(self):
        pass

    
    @staticmethod
    def tokenize_string(string: str) -> list:
        """
        simple function to tokenize string

        input:
        + string: self-explanatory
        
        output:
        + a list of tokens

        TODO: update this function
        """

        processed_string = string.replace(",", "")
        return processed_string.split(" ")

    @staticmethod
    def get_vocab(column: "pd.DataFrame column") -> (dict, dict):
        """
        Get full vocab from a column of you .csv data file

        input:
        + column: a column from your .csv file of your select

        output:
        + index2word: a dictionary where the keys are the indices, and the
        values are the words in the vocab
        + word2index: a dictionary where the values are the index, and the
        keys are the indices of the corresponding words
        """
        vocab = set([])

        def _get_vocab_from_row(tokens):
            for token in tokens:
                vocab.add(token)

        column.progress_apply(_get_vocab_from_row)

        vocab = list(vocab)
        index2word = {index + 1:vocab[index] for index in range(len(vocab))}
        word2index = {vocab[index]: index + 1 for index in range(len(vocab))}

        index2word[0] = "<pad>"
        word2index["<pad>"] = 0
        return index2word, word2index

    @staticmethod
    def encode_string(tokenized_string: list, word2index: dict) -> list:
        """
        input:
        + tokenized_string: a list of tokens
        + word2index: a dictionary where the keys are the words in the vocab
        and the values are their corresponding index

        output:
        + list of ints that represents the encoded sequence
        """
        encoded = [word2index[token] for token in tokenized_string]
        return encoded

    @staticmethod
    def encode_column(column, word2index):
        """
        input:
        + column: a column from your .csv file of your select
        + word2index: a dictionary where the keys are the words in the vocab
        and the values are their corresponding index

        output:
        + a pd.DataFrame column
        """
        def _encode_as_string(string: str):
            encoded_string = Preprocessing.encode_string(string, word2index)
            encoded_string = [str(item) for item in encoded_string]
            return ",".join(encoded_string)

        return column.progress_apply(_encode_as_string)


if __name__ == "__main__":
    parser = ArgumentParser(description='Process data')
    parser.add_argument("--data-path", default = "",
                        help = 'path to unprocessed data')
    
    parser.add_argument("--save-folder", default="",
                        help = "path to where to processed data will be saved")

    args = parser.parse_args()

    maybe_create_folder(args.save_folder)

    preprocessing = Preprocessing()

    df = pd.read_csv(args.data_path)

    # assert that the .csv file contains 2 column 'source' and 'dest'
    assert "source" in df.columns \
        and "dest" in df.columns, "the .csv file should contain 2 columns 'source' and 'dest'"
    
    df.source = df.source.progress_apply(lambda x: x.lower())
    df.dest = df.dest.progress_apply(lambda x: x.lower())


    # tokenize the sentences in the 'dest' and the 'source' columns
    df.source = df.source.progress_apply(preprocessing.tokenize_string)
    df.dest = df.dest.progress_apply(preprocessing.tokenize_string)

    # get the vocab
    index2word, word2index = preprocessing.get_vocab(df.dest)

    # save the vocab files
    index2word_file = os.path.join(args.save_folder, "index2word.json")
    if not os.path.exists(index2word_file):
        with open(index2word_file, "w") as f:
            json.dump(index2word, f)

    word2index_file = os.path.join(args.save_folder, "word2index.json")
    if not os.path.exists(word2index_file):
        with open(word2index_file, "w") as f:
            json.dump(word2index, f)

    df.source = preprocessing.encode_column(df.source, word2index)
    df.dest = preprocessing.encode_column(df.dest, word2index)

    df.to_csv(os.path.join(args.save_folder, "processed_data.csv"), index = False)