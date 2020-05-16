import torch
import yaml
import logging
import json
import os
from tqdm import tqdm
from model import Transformer
from data import TransformerDataset
from lib import CustomCrossEntropyLoss, maybe_create_folder
from preprocess import RawDataPrep

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ModelTrainer:
    def __init__(self,
                 config_path):
        
        # configurations
        CONFIG = ModelTrainer.get_config_yml(config_path)
        self.CONFIG = CONFIG

        # prepare data for training
        self.prepare_data_for_training()

        with open(os.path.join(CONFIG["vocab_path"], "index2word.json"), "r") as f:
            index2word = json.load(f)

        
        vocab_size = len(index2word)

        maybe_create_folder(CONFIG["ckpt_folder"])

        train_dataset = TransformerDataset(data_path = CONFIG["train_data_path"],
                                           max_len = CONFIG["max_len"],
                                           vocab_size = vocab_size)
        

        # train loader
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        shuffle = True,
                                                        batch_size = CONFIG["batch_size"],
                                                        pin_memory = True,
                                                        drop_last = True)
        

        # declare model to train
        device = torch.device(CONFIG["device"])

        self.model = Transformer(d_model = CONFIG["d_model"],
                                 nhead = CONFIG["nhead"],
                                 num_layers = CONFIG["num_layers"],
                                 dropout = CONFIG["dropout"],
                                 vocab_size = vocab_size,
                                 max_len = CONFIG["max_len"])

        # resume training from specified checkpoint
        self.resume_training()
        
        self.model.to(device)
        self.device = device

        # Adam optimizer
        # TODO: add learning rate scheduler like in the original paper
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": CONFIG["lr"]},
                {"params": self.model.decoder.parameters(), "lr": CONFIG["lr"]},
                {"params": self.model.wordemb.parameters(), "lr": CONFIG["lr"]},
            ]
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=CONFIG["step_size"], 
                                                         gamma=CONFIG["lr_decay"])


        self.criterion = CustomCrossEntropyLoss()


    def _train(self, epoch: int):
        # 1 epoch training

        train_pbar = tqdm(self.train_loader)
        train_pbar.desc = f'* Epoch {epoch+1}'

        tgt_mask = TransformerDataset.generate_square_subsequent_mask(self.CONFIG["max_len"]).to(self.device)

        for (source_sequences, inp_dest, tar_dest,
             source_key_padding_mask, dest_key_padding_mask, loss_mask) in train_pbar:

            source_sequences = source_sequences.to(self.device)
            inp_dest = inp_dest.to(self.device)
            tar_dest = tar_dest.to(self.device)
            source_key_padding_mask = source_key_padding_mask.to(self.device)
            dest_key_padding_mask = dest_key_padding_mask.to(self.device)
            loss_mask = loss_mask.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(source_sequences.transpose(0, 1), 
                                     inp_dest.transpose(0, 1),
                                     tgt_mask = tgt_mask,
                                     src_key_padding_mask = source_key_padding_mask,
                                     tgt_key_padding_mask = dest_key_padding_mask)
            
            predictions = predictions.permute(0, 2, 1)

            loss = self.criterion(predictions, tar_dest, loss_mask)
            loss.backward()

            self.optimizer.step()

            train_pbar.set_postfix({
                "train_loss": loss.item()
            })
            
    def train_model(self):
        self.model.train()

        for epoch in range(self.CONFIG["epochs"]):
            self._train(epoch)
            # adjust learning rate after 1 epoch
            self.scheduler.step()
        
        # save the model
        self.save_checkpoint()


    def resume_training(self):
        if os.path.exists(self.CONFIG["resume_path"]):
            state_dict = torch.load(os.path.join(self.CONFIG["ckpt_folder"], 
                                                 "best_checkpoint.pth.tar"))
            self.model.load_state_dict(state_dict)
            logging.info("Resume training from {}".format(self.CONFIG["resume_path"]))
        else:
            logging.info("Start training from scratch")

    
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), 
                   os.path.join(self.CONFIG["ckpt_folder"], "best_checkpoint.pth.tar"))

        with open(os.path.join(self.CONFIG["ckpt_folder"], "best_config.json"), "w") as f:
            json.dump(self.CONFIG, f)

    def prepare_data_for_training(self):
        raw_data_path = self.CONFIG["raw_data_path"]
        data_path = self.CONFIG["train_data_path"]
        vocab_path = self.CONFIG["vocab_path"]
        
        if not os.path.exists(data_path):
            logging.info("Data isn't yet available, preparing data")
            rawdataprep = RawDataPrep()
            df = rawdataprep.get_training_data(raw_data_path, vocab_path)
            df.to_csv(data_path, index=False)


    @staticmethod
    def get_config_yml(config_path):
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)



if __name__ == "__main__":
    model_trainer = ModelTrainer("./config/train.yml")
    model_trainer.train_model()