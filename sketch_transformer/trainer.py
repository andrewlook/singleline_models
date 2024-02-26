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

from .config import get_default_config
from .dataset import StrokesDataset
from .masks import create_masks
from .model import *
from .utils import CN

# from .sampler import Sampler

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

        print('='*60)
        print(f"RUN_ID: {self.run_id}\n")
        print(f"HYPERPARAMETERS:\n")
        print(json.dumps(hp.__dict__, indent=2))
        print('='*60 + '\n\n')

        self.models_dir = Path(models_dir)
        self.run_dir = self.models_dir / self.run_id
        if not os.path.isdir(self.run_dir):
            os.makedirs(self.run_dir)

        # Initialize step count, to be updated in the training loop
        self.total_steps = 0
        
        self.model = Model(hp=self.hp).to(self.device)
        self.loss = ReconstructionLoss()

        if self.use_wandb:
            wandb.watch(self.model, log="all", log_freq=10, log_graph=True)

        # store learning rate as state, so it can be modified by LR decay
        self.learning_rate = self.hp.lr
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        # `npz` file path is `data/quickdraw/[DATASET NAME].npz`
        base_path = Path(f"data/{hp.dataset_source}")
        path = base_path / f'{hp.dataset_name}.npz'
        # Load the numpy file
        dataset = np.load(str(path), encoding='latin1', allow_pickle=True)

        # Create training dataset
        self.train_dataset = StrokesDataset(dataset['train'], hp.max_seq_length)
        # Create validation dataset
        self.valid_dataset = StrokesDataset(dataset['valid'], hp.max_seq_length, self.train_dataset.scale)

        def collate_fn(batch, **kwargs):
            assert type(batch) == list
            # assert len(batch) == hp.batch_size

            all_data = []
            all_mask = []
            for data, mask in batch:
                assert data.shape[0] == hp.max_seq_length + 2
                assert data.shape[1] == 5
                assert len(data.shape) == 2
                ### NOTE: this line is different from RNN version, to ensure mask is same
                ### size as the input sequence
                assert mask.shape[0] == hp.max_seq_length + 2
                # assert mask.shape[0] == hp.max_seq_length + 1
                assert len(mask.shape) == 1
                # _data = data
                # if hp.use_random_scale:
                #     _data = random_scale(data, hp.random_scale_factor)
                # if hp.augment_stroke_prob > 0:
                #     _data = augment_strokes(_data, hp.augment_stroke_prob)
                all_data.append(data)
                all_mask.append(mask)
            # print(f"collate - batch: {len(batch)}, {batch[0][0].shape}, {batch[0][1].shape}")
            # print(f"collate - kwargs: {kwargs}")
            return torch.stack(all_data), torch.stack(all_mask)

        # Create training data loader
        self.train_loader = DataLoader(self.train_dataset, hp.batch_size, shuffle=True, collate_fn=collate_fn)
        # Create validation data loader
        self.valid_loader = DataLoader(self.valid_dataset, hp.batch_size)

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
            self.total_steps = idx + epoch * steps_per_epoch
            loss, _ = self.step(batch, is_training=True)
            self.log(dict(
                loss=loss,
                epoch=epoch,
                learning_rate=self.learning_rate))
        
        # update learning rate, if use_lr_decay is enabled
        if self.hp.use_lr_decay:
            if self.learning_rate > self.hp.min_lr:
                self.learning_rate *= self.hp.lr_decay
            self.optimizer = self.update_lr(self.optimizer, self.learning_rate)

    def update_lr(self, optimizer, lr):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
        
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
