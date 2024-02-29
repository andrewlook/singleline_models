# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/sketch_transformer/04_trainer.ipynb.

# %% auto 0
__all__ = ['hp', 'get_default_config', 'Trainer']

# %% ../../nbs/sketch_transformer/04_trainer.ipynb 4
import json
import math
import os
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..dataset import StrokesDataset, create_dataloaders
from ..utils import CN

from .masks import create_masks
from .model import *

# %% ../../nbs/sketch_transformer/04_trainer.ipynb 5
def get_default_config():
    C = CN()
    C.n_layer = 4
    C.n_head = 8
    C.d_model =  128
    C.d_ff = 512
    C.d_lowerdim = 256

    # these options must be filled in externally
    C.vocab_size = None
    C.block_size = None
    # dropout hyperparameters
    C.dropout_rate = 0.1


    C.max_seq_length = 250
    C.batch_size = 100

    C.blind_decoder_mask = True # if True, the decoder knows padding location of the input

    # TODO: just make this a path?
    C.dataset_source: str = 'look'
    C.dataset_name: str = 'epoch20240221_expanded10x_trainval'
    C.dataset_fname: str = 'data/look/epoch20240221_expanded10x_trainval.npz'
    # C.dataset_source: str = 'look'
    # C.dataset_name: str = 'look_i16__minn10_epsilon1'
    # C.dataset_fname: str = 'data/look/look_i16__minn10_epsilon1.npz'
    # data augmentation
    C.augment_stroke_prob = 0.1
    C.use_random_scale = True
    C.random_scale_factor = 0.15

    C.epochs = 50000
    C.lr = 1e-3
    C.use_lr_decay = True
    C.min_lr = 1e-5
    C.lr_decay = 0.9999
    
    return C

hp = get_default_config()
class Trainer():
    # Device configurations to pick the device to run the experiment
    device: str
    
    model: Model
    loss: ReconstructionLoss
    optimizer: optim.Adam
    # sampler: Sampler

    train_loader: DataLoader
    valid_loader: DataLoader
    train_dataset: StrokesDataset
    valid_dataset: StrokesDataset

    learning_rate: float
    best_val_loss: float = float('inf')

    def __init__(self,
                 hp: CN,
                 device="cuda",
                 models_dir="models",
                 use_wandb=False,
                 wandb_project='sketch-transformer',
                 wandb_entity='andrewlook'):
        self.hp = hp
        self.device = device
        self.use_wandb = use_wandb
        
        # create a unique run ID, to distinguish saved model checkpoints / sample images
        self.run_id = f"{math.floor(np.random.rand() * 1e6):07d}"
        if self.use_wandb:
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=hp.__dict__,
            )
            # use wandb's run ID, if available, so checkpoints match W&B's dashboard ID
            self.run_id = run.id
        self.models_dir = Path(models_dir)
        self.run_dir = self.models_dir / self.run_id
        if not os.path.isdir(self.run_dir):
            os.makedirs(self.run_dir)

        print('='*60)
        print(f"RUN_ID: {self.run_id}\n")
        print(f"HYPERPARAMETERS:\n")
        print(json.dumps(hp.__dict__, indent=2))
        print('='*60 + '\n\n')

        # Initialize step count, to be updated in the training loop
        self.total_steps = 0
        
        self.model = Model(hp=self.hp).to(self.device)
        self.loss = ReconstructionLoss()

        if self.use_wandb:
            wandb.watch(self.model, log="all", log_freq=10, log_graph=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=1000, num_training_steps=50000)

        # # store learning rate as state, so it can be modified by LR decay
        # self.learning_rate = self.hp.lr
        # self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        self.train_dataset, self.train_loader, self.valid_dataset, self.valid_loader = create_dataloaders(hp)

        # # Create sampler
        # self.sampler = Sampler(self.encoder, self.decoder)
        # # Pick 5 indices from the validation dataset, so the sampling can be compared across epochs
        # self.valid_idxs = [np.random.choice(len(self.valid_dataset)) for _ in range(5)]

    def save(self):
        torch.save(self.model.state_dict(), \
            Path(self.run_dir) / f'runid-{self.run_id}.pth')
        with open(Path(self.run_dir) / f'runid-{self.run_id}.json', 'w') as outfile:
            json.dump(self.hp.__dict__, outfile, indent=2)

    @staticmethod
    def load(self, **trainer_args):
        with open(Path(self.run_dir) / f'runid-{self.run_id}.json', 'r') as infile:
            saved_hp = json.load(infile)
        hp = get_default_config()
        hp.merge_from_dict(saved_hp)
        trainer = Trainer(hp=hp, **trainer_args)
        
        extra_args = {}
        if trainer.device != 'cuda':
            extra_args=dict(map_location=torch.device('cpu'))
        saved_model = torch.load(Path(self.run_dir) / f'runid-{self.run_id}.pth', **extra_args)

        trainer.model.load_state_dict(saved_model)
        return trainer
    
    def log(self, metrics):
        if self.use_wandb:
            wandb.log(metrics, step=self.total_steps)
        else:
            pass
            #pprint({'step': self.total_steps, **metrics})

    # def sample(self, epoch, display=False):
    #     orig_paths = []
    #     decoded_paths = []
    #     for idx in self.valid_idxs:
    #         orig_path = self.run_dir / f'runid-{self.run_id}_epoch-{epoch:05d}_sample-{idx:04d}_orig.png'
    #         decoded_path = self.run_dir / f'runid-{self.run_id}_epoch-{epoch:05d}_sample-{idx:04d}_decoded.png'

    #         # Randomly pick a sample from validation dataset to encoder
    #         data, *_ = self.valid_dataset[idx]
    #         self.sampler.plot(data, orig_path)

    #         # Add batch dimension and move it to device
    #         data_batched = data.unsqueeze(1).to(self.device)
    #         # Sample
    #         self.sampler.sample(data_batched, self.hp.temperature, decoded_path)

    #         if display:
    #             Image.open(orig_path).show()
    #             Image.open(decoded_path).show()
    #         orig_paths.append(orig_path)
    #         decoded_paths.append(decoded_path)
    #     return sorted(orig_paths), sorted(decoded_paths)   

    def step(self, batch: Any, is_training=False):
        self.model.train(is_training)

        data = batch[0].to(self.device)
        
        # hack: add 1 dimension
        inp = torch.cat([data, data[:, -1, :].unsqueeze(1)], dim=1)
        tar_inp = inp[:, :-1, ...]
        tar_real = inp[:, 1:, ...]

        enc_padding_mask, dec_padding_mask, dec_target_padding_mask, look_ahead_mask = create_masks(tar_inp, tar_inp, device=self.device)

        recon, _ = self.model(tar_inp, tar_inp, enc_padding_mask, dec_padding_mask, dec_target_padding_mask, look_ahead_mask)

        loss = self.loss(recon, tar_real)

        # Only if we are in training state
        if is_training:
            # Set `grad` to zero
            self.optimizer.zero_grad()
            
            # Compute gradients
            loss.backward()
            
            # # Clip gradients
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.hp.grad_clip)
            
            # Optimize
            self.optimizer.step()
        return loss.item(), data.shape[0]

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_items, total_loss = 0, 0
        with torch.no_grad():    
            for batch in iter(self.valid_loader):
                loss, batch_items = self.step(batch, is_training=False)

                total_loss += loss * batch_items
                total_items += batch_items
                
        avg_loss = total_loss / total_items
        self.log(dict(
            val_avg_loss=avg_loss,
            epoch=epoch))
        return avg_loss

    def train_one_epoch(self, epoch, parent_progressbar=None):
        steps_per_epoch = len(self.train_loader)
        for idx, batch in enumerate(progress_bar(iter(self.train_loader), parent=parent_progressbar)):
            self.scheduler.step()
            self.total_steps = idx + epoch * steps_per_epoch
            loss, _ = self.step(batch, is_training=True)
            self.log(dict(
                loss=loss,
                epoch=epoch,
                learning_rate=self.optimizer.param_groups[0]['lr']))
        
    #     # update learning rate, if use_lr_decay is enabled
    #     if self.hp.use_lr_decay:
    #         if self.learning_rate > self.hp.min_lr:
    #             self.learning_rate *= self.hp.lr_decay
    #         self.optimizer = self.update_lr(self.optimizer, self.learning_rate)

    # def update_lr(self, optimizer, lr):
    #     """Decay learning rate by a factor of lr_decay"""
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     return optimizer
        
    def train(self):
        mb = master_bar(range(self.hp.epochs))
        for epoch in mb:
            self.train_one_epoch(epoch=epoch, parent_progressbar=mb)
            val_avg_loss = self.validate_one_epoch(epoch)
            update_best_val = False
            if val_avg_loss < self.best_val_loss:
                self.best_val_loss = val_avg_loss
                update_best_val = True
                #if epoch % self.hp.save_every_n_epochs == 0:
                self.save()
                # self.sample()
            mb.write(f"Finished epoch {epoch}. Validation Loss: {val_avg_loss}{' (new best)' if update_best_val else ''}")

