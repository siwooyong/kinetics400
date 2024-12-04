import argparse

import os

import pandas as pd
import numpy as np

import torch

import timm

import lightning as L

from transformers.optimization import get_cosine_schedule_with_warmup

from src.dataset import CustomDataset
from src.model import CustomModel

import warnings
warnings.filterwarnings(action = "ignore")

class LightningModel(L.LightningModule):
    def __init__(self, args):
        super(LightningModel, self).__init__()
        self.args = args

        self.model = CustomModel(args)

        self.log_path = f'{args.save_dir}/log.txt'

        self.training_step_outputs = []
        self.validation_step_outputs = []

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    def training_step(self, batch, batch_idx):
        inputs, masks = batch

        outputs = self.model(inputs, masks)
        
        loss = outputs.loss
        
        self.training_step_outputs.append({"train_loss" : loss.cpu().detach()})
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, masks = batch
        
        outputs = self.model(inputs, masks)
        
        loss = outputs.loss

        self.validation_step_outputs.append({"test_loss" : loss.cpu().detach()})
        return loss

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        lr = self.optimizers().param_groups[0]['lr']
        
        train_loss = torch.stack([x["train_loss"] for x in self.training_step_outputs], dim = 0)
        train_loss = self.all_gather(train_loss).mean()
        train_loss = train_loss.detach().cpu().tolist()
        
        test_loss = torch.stack([x["test_loss"] for x in self.validation_step_outputs], dim = 0)
        test_loss = self.all_gather(test_loss).mean()
        test_loss = test_loss.detach().cpu().tolist()
        
        if self.global_rank == 0:
            log = f'epoch:{epoch}, lr:{lr}, train_loss:{train_loss:.6f}, test_loss:{test_loss:.6f}\n'
            self.log(self.args, log)

            if ((epoch+1) % self.args.save_frequency) == 0:
                save_path = self.args.save_dir + '/epoch:' + \
                    f'{epoch}'.zfill(3) + \
                    f'-train_loss:{train_loss:.6f}' + \
                    f'-test_loss:{test_loss:.6f}' + '.bin'
                torch.save(self.model.state_dict(), save_path)
                
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr = self.args.lr, 
            weight_decay = self.args.wd,
            betas = (self.args.betas1, self.args.betas2)
            )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(self.args.total_steps * self.args.warmup_ratio),
            num_training_steps = self.args.total_steps
            )
        return [optimizer], [{"scheduler" : scheduler, "interval" : "step", "frequency" : 1}]

    def log(self, args, message):
        print(message)
        with open(f'{args.save_dir}/log.txt', 'a+') as logger:
            logger.write(f'{message}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", type = str, default = "pretrained_weights")
    parser.add_argument("--model_name", type = str, default = "MCG-NJU/videomae-base")
    
    parser.add_argument("--mode", type = str, default = "self-supervised")
    
    parser.add_argument("--clip_length", type = int, default = 16)
    parser.add_argument("--sample_rate", type = int, default = 4)
    parser.add_argument("--shortest_edge", type = int, default = 224)
    
    parser.add_argument("--input_size", type = int, default = 224)
    parser.add_argument("--n_sample", type = int, default = -1)
    
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--n_worker", type = int, default = 16)
    parser.add_argument("--n_device", type = int, default = 8)
    
    parser.add_argument("--device", type = str, default = "cuda")
    
    parser.add_argument("--view", type = str, default = "single")
    
    parser.add_argument("--precision", type = str, default = "bf16-mixed")
    parser.add_argument("--dtype", type = str, default = "bfloat16")
    
    parser.add_argument("--strategy", type = str, default = "ddp")
    
    parser.add_argument("--n_epoch", type = int, default = 400)
    
    parser.add_argument("--lr", type = float, default = 5e-4)
    parser.add_argument("--wd", type = float, default = 5e-2)
    parser.add_argument("--warmup_ratio", type = float, default = 0.05)
    parser.add_argument("--betas1", type = float, default = 0.9)
    parser.add_argument("--betas2", type = float, default = 0.95)
    
    parser.add_argument("--save_frequency", type = int, default = 20)
    
    parser.add_argument("--num_sanity_val_steps", type = int, default = 0)

    parser.add_argument("--gradient_clip_val", type = float, default = 0.0)
    
    parser.add_argument("--hflip_prob", type = float, default = 0.5)
    
    parser.add_argument("--attn_implementation", type = str, default = "sdpa")
    parser.add_argument("--hidden_dropout_prob", type = float, default = 0.0)
    parser.add_argument("--attention_probs_dropout_prob", type = float, default = 0.0)

    parser.add_argument("--patch_size", type = int, default = 16)
    parser.add_argument("--tubelet_size", type = int, default = 2)
    parser.add_argument("--mask_ratio", type = float, default = 0.9)
    parser.add_argument("--norm_pix_loss", action = 'store_true')
    
    args = parser.parse_args()
    
    train = pd.read_csv(f"data/Kinetics-400/kinetics400_train_list_videos.txt", sep = " ", header = None)
    val = pd.read_csv(f"data/Kinetics-400/kinetics400_val_list_videos.txt", sep = " ", header = None)
    
    if args.n_sample > 0:
        train = train[:args.n_sample]
        val = val[:args.n_sample]
    
    train_dataset = CustomDataset(args, df = train, split = "train", is_training = True)
    test_dataset = CustomDataset(args, df = val, split = "val", is_training = False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        num_workers = args.n_worker,
        shuffle = True,
        drop_last = True,
        pin_memory = True,
        persistent_workers = args.n_worker>0,
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        num_workers = args.n_worker,
        shuffle = False,
        drop_last = True,
        pin_memory = True,
        persistent_workers = args.n_worker>0,
        ) 
    
    args.total_steps = int(len(train_dataset)*args.n_epoch/(args.batch_size*args.n_device))

    model = LightningModel(args)
    
    trainer = L.Trainer(
        accelerator = "gpu",
        devices = args.n_device,
        precision = args.precision,
        max_epochs = args.n_epoch,
        logger = False,
        num_sanity_val_steps = args.num_sanity_val_steps,
        enable_checkpointing = False,
        strategy = args.strategy,
        gradient_clip_val = args.gradient_clip_val,
    )
    trainer.fit(model, train_loader, test_loader)