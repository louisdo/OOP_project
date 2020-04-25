import json
import os
import torch 
from preprocess.preprocess import Preprocessing
from model import Transformer

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

        self.model = Transformer(d_model = config["d_model"],
                                 nhead = config["nhead"],
                                 num_layers = config["num_layers"],
                                 dropout = config["dropout"],
                                 vocab_size = config["vocab_size"] + 2,
                                 max_len = config["max_len"])

        state_dict = torch.load(checkpoint_file)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        word2index_file = os.path.join(vocab_folder, "word2index.json")
        index2word_file = os.path.join(vocab_folder, "index2word.json")

        with open(word2index_file, "r") as f:
            word2index = json.load(f)
            self.word2index = {item[0]:int(item[1]) for item in word2index.items()}

        with open(index2word_file, "r") as f:
            index2word = json.load(f)
            self.index2word = {int(item[0]):item[1] for item in index2word.items()}

        self.sos = torch.tensor([[len(self.word2index)]]).to(self.device).long()

        self.softmax = torch.nn.Softmax(dim=1)
        
    def predict(self, source: str):
        """
        input:
        + source: the source sentence

        output:
        + the predicted sequence
        """
        tokenized_source = Preprocessing.tokenize_string(source)

        encoded_source = [self.word2index[token] for token in tokenized_source]
        
        encoded_source = torch.tensor(encoded_source).unsqueeze(0).to(self.device).transpose(0, 1)

        output = self.sos.detach().clone()

        count = 0

        while output[0][-1].item() != len(self.word2index) + 1 and count <=100:
            model_output = self.model(encoded_source, output.transpose(0, 1))[:,-1:,:]
            model_output = self.softmax(model_output.squeeze(0))

            next_word = torch.argmax(model_output, dim=1)[-1].item()
            output = torch.cat([output, torch.tensor([[next_word]])], dim=1)
            count += 1

        output = output.squeeze(0)

        result = []
        for ind in output:
            index = ind.item()
            if index<=12:
                result.append(self.index2word[index])

        return " ".join(result)

    def __call__(self, source):
        return self.predict(source)