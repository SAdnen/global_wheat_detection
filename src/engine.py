import sys
import math
import torch
from .metrics import map_score
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from .model import Models
from src.metrics import MetricLogger
from src.utils import mixup_images, merge_targets

from time import time


class Detector(object):
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.model = Models().get_model(cfg["network"]) # cfg.network
        self.model.to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=0.00001)
        self.lr_scheduler = OneCycleLR(self.optimizer,
                                       max_lr=1e-4,
                                       epochs=cfg["nepochs"],
                                       steps_per_epoch=169,  # len(dataloader)/accumulations
                                       div_factor=25,  # for initial lr, default: 25
                                       final_div_factor=1e3,  # for final lr, default: 1e4
                                       )

    def fit(self, data_loader, accumulation_steps=4, wandb=None):
        self.model.train()
        #     metric_logger = utils.MetricLogger(delimiter="  ")
        #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        avg_loss = MetricLogger('scalar')
        total_loss = MetricLogger('dict')
        #lr_log = MetricLogger('list')

        self.optimizer.zero_grad()
        device = self.device

        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.detach().item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            losses.backward()
            if (i+1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    wandb.log({"lr": self.lr_scheduler.get_last_lr()[0]})
                    #lr_log.update(self.lr_scheduler.get_last_lr())


            print(f"Train iteration: [{i+1}/{len(data_loader)}]\r", end="")
            avg_loss.update(loss_value)
            total_loss.update(loss_dict)

            # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print()
        #print(loss_dict)
        return {"train_avg_loss": avg_loss.avg}, total_loss.avg


    def mixup_fit(self, data_loader, accumulation_steps=4, wandb=None):
        self.model.train()
        torch.cuda.empty_cache()
        #     metric_logger = utils.MetricLogger(delimiter="  ")
        #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        avg_loss = MetricLogger('scalar')
        total_loss = MetricLogger('dict')
        #lr_log = MetricLogger('list')

        self.optimizer.zero_grad()
        device = self.device

        for i, (batch1, batch2) in enumerate(data_loader):
            images1, targets1 = batch1
            images2, targets2 = batch2
            images = mixup_images(images1, images2)
            targets = merge_targets(targets1, targets2)
            del images1, images2, targets1, targets2, batch1, batch2

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.detach().item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            losses.backward()
            if (i+1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    #lr_log.update(self.lr_scheduler.get_last_lr())


            print(f"Train iteration: [{i+1}/{674}]\r", end="")
            avg_loss.update(loss_value)
            total_loss.update(loss_dict)

            # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print()
        #print(loss_dict)
        return {"train_avg_loss": avg_loss.avg}, total_loss.avg


    def evaluate(self, val_dataloader):
        device = self.device
        torch.cuda.empty_cache()
        # self.model.to(device)
        self.model.eval()
        mAp_logger = MetricLogger('list')
        with torch.no_grad():
            for (j, batch) in enumerate(val_dataloader):
                print(f"Validation: [{j+1}/{len(val_dataloader)}]\r", end="")
                images, targets = batch
                del batch
                images = [img.to(device) for img in images]
                # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = self.model(images)#, targets)
                for i, pred in enumerate(predictions):
                    probas = pred["scores"].detach().cpu().numpy()
                    mask = probas > 0.6
                    preds = pred["boxes"].detach().cpu().numpy()[mask]
                    gts = targets[i]["boxes"].detach().cpu().numpy()
                    score, scores = map_score(gts, preds, thresholds=[.5, .55, .6, .65, .7, .75])
                    mAp_logger.update(scores)
            print()
        return {"validation_mAP_score": mAp_logger.avg}

    def get_checkpoint(self):
        self.model.eval()
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        checkpoint = {'model_state_dict': model_state,
                      'optimizer_state_dict': optimizer_state
                      }
        # if self.lr_scheduler:
        #     scheduler_state = self.lr_scheduler.state_dict()
        #     checkpoint['lr_scheduler_state_dict'] = scheduler_state

        return checkpoint

    def load_checkpoint(self, checkpoint):
        self.model.eval()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # if self.lr_scheduler:
        #     self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
