import torch
import yaml
import logging
import json
import os
from tqdm import tqdm
from model import Transformer
from data import TransformerDataset
from lib import CustomCrossEntropyLoss, maybe_create_folder

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ModelTrainer:
    def __init__(self,
                 config_path):
        
        # configurations
        CONFIG = ModelTrainer.get_config_yml(config_path)
        self.CONFIG = CONFIG
        
        real_vocab_size = CONFIG["vocab_size"] + 2

        maybe_create_folder(CONFIG["ckpt_folder"])

        train_dataset = TransformerDataset(data_path = CONFIG["train_data_path"],
                                           max_len = CONFIG["max_len"],
                                           vocab_size = real_vocab_size - 2)
        
        """test_dataset = TransformerDataset(data_path = CONFIG["test_data_path"],
                                          max_len = CONFIG["max_len"],
                                          vocab_size = real_vocab_size)"""

        # train loader and test loader
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        shuffle = True,
                                                        batch_size = CONFIG["batch_size"],
                                                        pin_memory = True,
                                                        drop_last = True)
        
        """self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       shuffle = False,
                                                       batch_size = CONFIG["batch_size"],
                                                       pin_memory = True,
                                                       drop_last = False)"""

        # declare model to train
        #device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        device = torch.device("cpu")
        self.model = Transformer(d_model = CONFIG["d_model"],
                                 nhead = CONFIG["nhead"],
                                 num_layers = CONFIG["num_layers"],
                                 dropout = CONFIG["dropout"],
                                 vocab_size = real_vocab_size,
                                 max_len = CONFIG["max_len"])

        self.resume_training()
        self.model.to(device)
        self.device = device

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": CONFIG["lr"]},
                {"params": self.model.decoder.parameters(), "lr": CONFIG["lr"]},
                {"params": self.model.wordemb.parameters(), "lr": CONFIG["lr"]},
            ]
        )

        self.criterion = CustomCrossEntropyLoss()


    def _train(self, epoch):
        # 1 epoch training

        train_pbar = tqdm(self.train_loader)
        train_pbar.desc = f'* Epoch {epoch+1}'

        for batch_idx, (source_sequences, 
                        inp_dest, tar_dest) in enumerate(train_pbar):

            source_sequences = source_sequences.to(self.device)
            inp_dest = inp_dest.to(self.device)
            tar_dest = tar_dest.to(self.device)

            source_key_padding_mask = ModelTrainer.generate_mask(source_sequences).to(self.device)
            dest_key_padding_mask = ModelTrainer.generate_mask(inp_dest).to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(source_sequences.transpose(0, 1), 
                                     inp_dest.transpose(0, 1),
                                     src_key_padding_mask = source_key_padding_mask,
                                     tgt_key_padding_mask = dest_key_padding_mask)
            
            predictions = predictions.permute(0, 2, 1)

            loss = self.criterion(predictions, tar_dest, torch.bitwise_not(dest_key_padding_mask))
            loss.backward()

            self.optimizer.step()

            train_pbar.set_postfix({
                "train_loss": loss.item()
            })
            
    def train_model(self):
        self.model.train()

        for epoch in range(self.CONFIG["epochs"]):
            self._train(epoch)
        
        # save the model
        self.save_checkpoint()


    def resume_training(self):
        if os.path.exists(self.CONFIG["resume_path"]):
            state_dict = torch.load(os.path.join(self.CONFIG["ckpt_folder"], 
                                                 "best_checkpoint.pth.tar"))
            self.model.load_state_dict(state_dict)
            print("Resume training from {}".format(self.CONFIG["resume_path"]))
        else:
            print("Start training from scratch")

    
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), 
                   os.path.join(self.CONFIG["ckpt_folder"], "best_checkpoint.pth.tar"))

        with open(os.path.join(self.CONFIG["ckpt_folder"], "best_config.json"), "w") as f:
            json.dump(self.CONFIG, f)


    @staticmethod
    def generate_mask(padded_seq):
        mask = padded_seq == 0
        return mask.bool()


    @staticmethod
    def get_config_yml(config_path):
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)



if __name__ == "__main__":
    model_trainer = ModelTrainer("./config/train.yml")
    model_trainer.train_model()