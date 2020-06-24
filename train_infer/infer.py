import json
import os
import torch
from pyvi import ViTokenizer
from model import Transformer
from preprocess import RawDataPrep
from lib import Utils

class TransformerInference:
    """
    Model Inference
    """
    def __init__(self, 
                 checkpoint_folder: str, 
                 vocab_folder: str, 
                 device: str):
        """
        input:
        + checkpoint_folder: path to folder where the checkpoint and the training config are saved,
        the folder should contain 'best_checkpoint.pth.tar' and 'best_config.json'
        + vocab_folder: path to folder where the vocab files are saved,
        the folder should contain 'word2index.json' and 'index2word.json'
        """

        config_file = os.path.join(checkpoint_folder, "best_config.json")
        checkpoint_file = os.path.join(checkpoint_folder, "best_checkpoint.pth.tar")

        with open(config_file, "r") as f:
            config = json.load(f)

        self.device = torch.device(device)

        index2word, word2index = Utils.load_vocab(vocab_folder = vocab_folder)
        self.word2index = {item[0]:int(item[1]) for item in word2index.items()}
        self.index2word = {int(item[0]):item[1] for item in index2word.items()}

        self.patterns = RawDataPrep.get_patterns()

        self.sos = torch.tensor([[self.word2index["<sos>"]]]).to(self.device).long()

        real_vocab_size = len(self.index2word)

        self.model = Transformer(d_model = config["d_model"],
                                 nhead = config["nhead"],
                                 num_layers = config["num_layers"],
                                 dropout = config["dropout"],
                                 vocab_size = real_vocab_size,
                                 max_len = config["max_len"])
        Utils.load_model(self.model, checkpoint_file)

        self.config = config
        self.model = self.model.to(self.device)
        self.model.eval()

        self.softmax = torch.nn.Softmax(dim=1)
        
    def predict(self, source: str):
        """
        input:
        + source: the source sentence

        output:
        + the predicted sequence
        """
        
        processed_source, replaced_terms = RawDataPrep.process_sentence(source, self.patterns)
        tokenized_source = ViTokenizer.tokenize(processed_source).split(" ")

        encoded_source = Utils.encode_line(tokenized_source, self.word2index, type_converter=int)
        encoded_source = torch.tensor(encoded_source).unsqueeze(0).to(self.device).transpose(0, 1)

        output = self.sos.detach().clone()

        count = 0

        while output[0][-1].item() != self.word2index["<eos>"] and count <=100:
            model_output = self.model(encoded_source, output.transpose(0, 1))[:,-1:,:]
            model_output = self.softmax(model_output.squeeze(0))

            next_word = torch.argmax(model_output, dim=1)[-1].item()
            output = torch.cat([output, torch.tensor([[next_word]])], dim=1)
            count += 1

        output = output.squeeze(0)

        result = []
        for ind in output:
            index = ind.item()
            if 0 < index < len(self.index2word):
                word = self.index2word[index]
                if word in replaced_terms:
                    word = replaced_terms[word]
                result.append(word)

        result = result[1:-1]

        return " ".join(result).replace("_", " ").replace(" , ",", ").strip()

    def __call__(self, source: str):
        return self.predict(source)