import os
import errno
import torch
import yaml
import logging
import json


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def maybe_create_folder(folder: str):
        # create folder if it doesn't exist
        try:
            os.makedirs(folder)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    @staticmethod
    def encode_line(sentence, word2index, type_converter = str):
        encoded_sentence = [word2index["<sos>"]] + [word2index[word] for word in sentence] + [word2index["<eos>"]]
        return [type_converter(item) for item in encoded_sentence]


    @staticmethod
    def load_model(model, path):
        assert os.path.exists(path), "Cannot load model as '{}' does not exist"
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def get_config_yml(config_path):
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)

    @staticmethod
    def change_device(tensors, device):
        for index in range(len(tensors)):
            tensors[index] = tensors[index].to(device)

        if len(tensors) == 1:
            return tensors[0]
        return tensors

    @staticmethod
    def get_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        return logger

    @staticmethod
    def load_vocab(vocab_folder):
        word2index_path = os.path.join(vocab_folder, "word2index.json")
        index2word_path = os.path.join(vocab_folder, "index2word.json")

        for path in [word2index_path, index2word_path]:
            assert os.path.exists(path), "'{}' does not exist".format(path)

        with open(word2index_path, "r") as f:
            word2index = json.load(f)

        with open(index2word_path, "r") as f:
            index2word = json.load(f)

        return index2word, word2index

    @staticmethod
    def save_vocab(vocab_folder, index2word, word2index):
        Utils.maybe_create_folder(vocab_folder)

        word2index_path = os.path.join(vocab_folder, "word2index.json")
        index2word_path = os.path.join(vocab_folder, "index2word.json")

        with open(index2word_path, "w") as f:
            json.dump(index2word, f)

        with open(word2index_path, "w") as f:
            json.dump(word2index, f)

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


    @staticmethod
    def get_padded_sequence(sequence, max_len):
        padded_sequence = torch.zeros((max_len))
        padded_sequence[:len(sequence)] += sequence

        return padded_sequence