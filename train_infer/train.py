import torch
import yaml
import logging
import json
import os
import sys
sys.path.append("../")
from tqdm import tqdm
from model import Transformer
from data import TransformerDataset
from lib import CustomCrossEntropyLoss, Utils
from preprocess import RawDataPrep

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ModelTrainer:
    def __init__(self,
                 config_path):
        
        # configurations
        CONFIG = Utils.get_config_yml(config_path)
        self.CONFIG = CONFIG

        # get logger
        self.logger = Utils.get_logger()

        # prepare data for training
        self._prepare_data_for_training()

        with open(os.path.join(CONFIG["vocab_path"], "index2word.json"), "r") as f:
            index2word = json.load(f)

        
        vocab_size = len(index2word)

        Utils.maybe_create_folder(CONFIG["ckpt_folder"])

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
        self._resume_training()
        
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


    def _train_one_batch(self, batch_input):
        source_sequences, inp_dest, tar_dest, source_key_padding_mask, \
        dest_key_padding_mask, loss_mask, tgt_mask = batch_input

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

        return loss.item()

        

    def _train_one_epoch(self, epoch: int):
        # 1 epoch training

        train_progress_bar = tqdm(self.train_loader, desc = f"* Epoch {epoch+1}")

        tgt_mask = Utils.generate_square_subsequent_mask(self.CONFIG["max_len"]).to(self.device)

        for batch_tensors in train_progress_bar:
            batch_tensors = Utils.change_device(device = self.device, 
                                                tensors = batch_tensors)

            batch_tensors = list(batch_tensors) + [tgt_mask]
            loss_value = self._train_one_batch(batch_input = batch_tensors)

            train_progress_bar.set_postfix({
                "train_loss": loss_value
            })
            
    def train_model(self):
        self.model.train()

        for epoch in range(self.CONFIG["epochs"]):
            self._train_one_epoch(epoch)
            # adjust learning rate after 1 epoch
            self.scheduler.step()
        
        # save the model
        self._save_checkpoint()


    def _resume_training(self):
        if os.path.exists(self.CONFIG["resume_path"]):
            Utils.load_model(self.model, self.CONFIG["resume_path"])
            self.logger.info("Resume training from {}".format(self.CONFIG["resume_path"]))
        else:
            self.logger.info("Start training from scratch")

    
    def _save_checkpoint(self):
        model_save_path = os.path.join(self.CONFIG["ckpt_folder"], "best_checkpoint.pth.tar")
        Utils.save_model(self.model, model_save_path)

        with open(os.path.join(self.CONFIG["ckpt_folder"], "best_config.json"), "w") as f:
            json.dump(self.CONFIG, f)


    def _prepare_data_for_training(self):
        raw_data_path = self.CONFIG["raw_data_path"]
        data_path = self.CONFIG["train_data_path"]
        vocab_path = self.CONFIG["vocab_path"]
        
        if not os.path.exists(data_path):
            self.logger.info("Data isn't yet available, preparing data")
            rawdataprep = RawDataPrep()
            df = rawdataprep.get_training_data(raw_data_path, vocab_path)
            df.to_csv(data_path, index=False)



if __name__ == "__main__":
    model_trainer = ModelTrainer("../config/train.yml")
    model_trainer.train_model()