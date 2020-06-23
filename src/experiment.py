import pandas as pd
import random
import os
import torch
import numpy as np
from src.engine import Detector
from src.config import path_settings
from src.utils import Dloaders
from time import time


class Experiment:
    def __init__(self, config):
        self.cfg = config
        self.wandb = None
        self.initial_epoch = 0
        self.nepoch = self.cfg["nepochs"]
        self.checkpoint_path = path_settings["checkpoint"]

        self.seed_everything()
        self._load_dataloaders(self.cfg)
        self.runner = Detector(config)

    def attach_wandb(self, wandb):
        self.wandb = wandb
        # self.wandb.init(project=self.cfg["project"],
        #            id=self.cfg["id"],
        #            group=self.cfg["group"], # self.KFold_number
        #            config=self.cfg,
        #            resume="allow",
        #            )
        print("WandB attached to Experiment")

    def seed_everything(self):
        seed = self.cfg["seed"]
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _load_dataloaders(self, cfg):
        dataloaders = Dloaders().get_dataloaders(0, cfg)
        self.train_dataloader = dataloaders["train"]
        self.valid_dataloader = dataloaders["valid"]
        self.test_dataloader = dataloaders["test"]

    def _step(self, epoch):
        current_epoch = self.initial_epoch + epoch
        last_epoch = self.initial_epoch + self.nepoch
        print(f"Epoch[{current_epoch+1}/{last_epoch}]:")
        start = time()
        train_avg_loss, train_losses = self.runner.fit(self.train_dataloader,
                                                        accumulation_steps=self.cfg['accumulation_steps'],
                                                        wandb=self.wandb)
        val_score = self.runner.evaluate(self.valid_dataloader)
        duration = (time() - start)/60

        if self.wandb:
            self.wandb.log(train_avg_loss, commit=False)
            self.wandb.log(train_losses, commit=False)
            self.wandb.log(val_score, commit=True)
            # TODO log roc_curve
        self.save_checkpoint(current_epoch, score=val_score['validation_mAP_score'])

        print(f"Train: loss {train_avg_loss['train_avg_loss']:.4f}  {duration:.2f} minutes")
        print(f"Valid: mAP@[0.5 : 0.75, 0.05]: {val_score['validation_mAP_score']:.4f}")

    def run(self,):
        for i in range(self.nepoch):
            self._step(i)

    def lr_finder(self, ):
        self.runner.lr_finder(self.train_dataloader)

    def save_checkpoint(self, epoch, score=0):
        # best_score = 1
        checkpoint = self.runner.get_checkpoint()
        checkpoint["epoch"] = epoch
        checkpoint["score"] = score
        checkpoint_name = self.cfg["id"] + f"_epoch_{epoch}_.pth"
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        # if score>best_score:
        #     checkpoint_name = self.cfg["id"] + f"_epoch_{epoch}_bestscore_.pth"
        #     checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
        #     torch.save(checkpoint, checkpoint_path)
        # TODO add condition for wandb to save checkpoint/best_checkpoint
        last_epoch = self.initial_epoch + self.nepoch - 1
        if epoch == last_epoch and self.wandb:
            self.wandb.save(checkpoint_path)

    def load_checkpoint(self):
        checkpoint_name = self.cfg["checkpoint"]
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        self.runner.load_checkpoint(checkpoint)
        self.initial_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint {checkpoint_name} loaded!")

    def get_submission(self, from_checkpoint=False):
        if from_checkpoint:
            self.load_checkpoint()
            print("Predicting using model state from checkpoint!")
        else:
            print("Predicting using current model state!")
        paths, predictions = self.runner.predict(self.test_dataloader)
        submission_df = pd.DataFrame()
        submission_df["Id"] = paths
        submission_df["Label"] = predictions
        return NotImplementedError
