# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/sketch_rnn/04_trainer.ipynb.

# %% auto 0
__all__ = ['SketchRNNModel', 'Trainer']

# %% ../../nbs/sketch_rnn/04_trainer.ipynb 4
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

from ..dataset import StrokesDataset, create_dataloaders
from .sampler import Sampler

# %% ../../nbs/sketch_rnn/04_trainer.ipynb 5
from ..utils import CN
from ..lstm.all import LSTM_BUILTIN, LSTM_RNNLIB
from .layers import BivariateGaussianMixture, DecoderRNN, EncoderRNN, KLDivLoss, ReconstructionLoss

# %% ../../nbs/sketch_rnn/04_trainer.ipynb 6
class SketchRNNModel(nn.Module):

    def __init__(self, hp, device="cuda"):
        super().__init__()
        self.hp = hp
        self.device = device
        self.encoder = EncoderRNN(
            self.hp.d_z,
            self.hp.enc_hidden_size,
            use_recurrent_dropout=self.hp.use_recurrent_dropout,
            r_dropout_prob=self.hp.r_dropout_prob,
            use_layer_norm=self.hp.use_layer_norm,
            layer_norm_learnable=self.hp.layer_norm_learnable,
            lstm_impl=self.hp.lstm_impl,
        ).to(self.device)
        self.decoder = DecoderRNN(
            self.hp.d_z,
            self.hp.dec_hidden_size,
            self.hp.n_distributions,
            use_recurrent_dropout=self.hp.use_recurrent_dropout,
            r_dropout_prob=self.hp.r_dropout_prob,
            use_layer_norm=self.hp.use_layer_norm,
            layer_norm_learnable=self.hp.layer_norm_learnable,
            lstm_impl=self.hp.lstm_impl,
        ).to(self.device)
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.architecture = 'Pytorch-SketchRNN'

        C.dataset_source: str = 'look'
        C.dataset_name: str = 'look_i16__minn10_epsilon1'
        C.dataset_fname: str = 'data/look/look_i16__minn10_epsilon1.npz'

        # C.dataset_source = 'look'
        # C.dataset_name = 'epoch20240221_expanded10x_trainval'
        # C.dataset_fname = 'data/look/epoch20240221_expanded10x_trainval.npz'
        
        # data augmentation
        C.augment_stroke_prob = 0.1
        C.use_random_scale = True
        C.random_scale_factor = 0.15

        # duration of training run
        C.epochs = 50000
        # how often to compute validation metrics / persist / sample
        C.save_every_n_epochs = 100
        # validate_every_n_epochs = 2

        # adaptive learning rate
        C.lr = 1e-3
        C.use_lr_decay = False
        C.min_lr = 1e-5
        C.lr_decay = 0.9999

        # recurrent dropout
        C.use_recurrent_dropout = False
        C.r_dropout_prob = 0.1

        # layer normalization
        C.use_layer_norm = True
        C.layer_norm_learnable = False

        # lstm_impl = LSTM_BUILTIN
        C.lstm_impl = LSTM_RNNLIB
        
        # Encoder and decoder sizes
        C.enc_hidden_size = 256
        C.dec_hidden_size = 512

        # Batch size
        C.batch_size = 100

        # Number of features in $z$
        C.d_z = 128
        # Number of distributions in the mixture, $M$
        C.n_distributions = 20

        # Weight of KL divergence loss, $w_{KL}$
        C.kl_div_loss_weight = 0.5
        # decaying weight of KL loss
        C.use_eta = False
        C.eta_min = 1e-2
        C.eta_R = 0.99995

        # Gradient clipping
        C.grad_clip = 1.
        # Temperature $\tau$ for sampling
        C.temperature = 0.4

        # Filter out stroke sequences longer than $200$
        C.max_seq_length = 200
        return C

    def sample(self, data: torch.Tensor, temperature: float):
        # $N_{max}$
        longest_seq_len = len(data)

        # Get $z$ from the encoder
        z, _, _ = self.encoder(data)

        # Start-of-sequence stroke is $(0, 0, 1, 0, 0)$
        s = data.new_tensor([0, 0, 1, 0, 0])
        seq = [s]
        # Initial decoder is `None`.
        # The decoder will initialize it to $[h_0; c_0] = \tanh(W_{z}z + b_z)$
        state = None

        # We don't need gradients
        with torch.no_grad():
            # Sample $N_{max}$ strokes
            for i in range(longest_seq_len):
                # $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ is the input to the decoder
                data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
                # Get $\Pi$, $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$,
                # $q$ and the next state from the decoder
                dist, q_logits, state = self.decoder(data, z, state)
                # Sample a stroke
                s = self._sample_step(dist, q_logits, temperature)
                # Add the new stroke to the sequence of strokes
                seq.append(s)
                # Stop sampling if $p_3 = 1$. This indicates that sketching has stopped
                if s[4] == 1:
                    break

        # Create a PyTorch tensor of the sequence of strokes
        seq = torch.stack(seq)
        return seq

    @staticmethod
    def _sample_step(dist: BivariateGaussianMixture, q_logits: torch.Tensor, temperature: float):
        # Set temperature $\tau$ for sampling. This is implemented in class `BivariateGaussianMixture`.
        dist.set_temperature(temperature)
        # Get temperature adjusted $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        pi, mix = dist.get_distribution()
        # Sample from $\Pi$ the index of the distribution to use from the mixture
        idx = pi.sample()[0, 0]

        # Create categorical distribution $q$ with log-probabilities `q_logits` or $\hat{q}$
        q = torch.distributions.Categorical(logits=q_logits / temperature)
        # Sample from $q$
        q_idx = q.sample()[0, 0]

        # Sample from the normal distributions in the mixture and pick the one indexed by `idx`
        xy = mix.sample()[0, 0, idx]

        # Create an empty stroke $(\Delta x, \Delta y, q_1, q_2, q_3)$
        stroke = q_logits.new_zeros(5)
        # Set $\Delta x, \Delta y$
        stroke[:2] = xy
        # Set $q_1, q_2, q_3$
        stroke[q_idx + 2] = 1
        #
        return stroke

# %% ../../nbs/sketch_rnn/04_trainer.ipynb 7
class Trainer():
    # Device configurations to pick the device to run the experiment
    device: str
    
    model: SketchRNNModel
    encoder: EncoderRNN
    decoder: DecoderRNN
    optimizer: optim.Adam
    sampler: Sampler

    train_loader: DataLoader
    valid_loader: DataLoader
    train_dataset: StrokesDataset
    valid_dataset: StrokesDataset

    kl_div_loss = KLDivLoss()
    reconstruction_loss = ReconstructionLoss()

    learning_rate: float
    best_val_loss: float = float('inf')

    def __init__(self,
                 model: SketchRNNModel,
                 hp: CN,
                 device="cuda",
                 models_dir="models",
                 use_wandb=False,
                 wandb_project='sketchrnn-pytorch',
                 wandb_entity='andrewlook'):
        self.model = model

        # TODO: remove these once I finish refactor
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

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

        if self.use_wandb:
            wandb.watch((self.encoder, self.decoder), log="all", log_freq=10, log_graph=True)

        # store learning rate as state, so it can be modified by LR decay
        self.learning_rate = self.hp.lr
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), self.learning_rate)

        self.eta_step = self.hp.eta_min if self.hp.use_eta else 1

        self.train_dataset, self.train_loader, self.valid_dataset, self.valid_loader = create_dataloaders(hp)
        
        # Pick 5 indices from the validation dataset, so the sampling can be compared across epochs
        self.valid_idxs = [np.random.choice(len(self.valid_dataset)) for _ in range(5)]

    def save(self, epoch):
        torch.save(self.encoder.state_dict(), \
            Path(self.run_dir) / f'runid-{self.run_id}_epoch-{epoch:05d}_encoderRNN.pth')
        torch.save(self.decoder.state_dict(), \
            Path(self.run_dir) / f'runid-{self.run_id}_epoch-{epoch:05d}_decoderRNN.pth')

    def load(self, epoch):
        extra_args = {}
        if self.device != 'cuda':
            extra_args=dict(map_location=torch.device('cpu'))
        saved_encoder = torch.load(Path(self.run_dir) / f'runid-{self.run_id}_epoch-{epoch:05d}_encoderRNN.pth', **extra_args)
        saved_decoder = torch.load(Path(self.run_dir) / f'runid-{self.run_id}_epoch-{epoch:05d}_decoderRNN.pth', **extra_args)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)
    
    def log(self, metrics):
        if self.use_wandb:
            wandb.log(metrics, step=self.total_steps)
        else:
            pass
            #pprint({'step': self.total_steps, **metrics})

    def sample(self, epoch, display=False):
        orig_paths = []
        decoded_paths = []
        for idx in self.valid_idxs:
            orig_path = self.run_dir / f'runid-{self.run_id}_epoch-{epoch:05d}_sample-{idx:04d}_orig.png'
            decoded_path = self.run_dir / f'runid-{self.run_id}_epoch-{epoch:05d}_sample-{idx:04d}_decoded.png'

            # Randomly pick a sample from validation dataset to encoder
            data, *_ = self.valid_dataset[idx]
            Sampler.plot(data, orig_path)

            # Add batch dimension and move it to device
            data_batched = data.unsqueeze(1).to(self.device)
            # Sample
            sampled_seq = self.model.sample(data_batched, self.hp.temperature)
            Sampler.plot(sampled_seq, decoded_path)

            if display:
                Image.open(orig_path).show()
                Image.open(decoded_path).show()
            orig_paths.append(orig_path)
            decoded_paths.append(decoded_path)
        return sorted(orig_paths), sorted(decoded_paths)   

    def step(self, batch: Any, is_training=False):
        self.encoder.train(is_training)
        self.decoder.train(is_training)

        # Move `data` and `mask` to device and swap the sequence and batch dimensions.
        # `data` will have shape `[seq_len, batch_size, 5]` and
        # `mask` will have shape `[seq_len, batch_size]`.
        data = batch[0].to(self.device).transpose(0, 1)
        mask = batch[1].to(self.device).transpose(0, 1)
        batch_items = len(data)

        # print(f"Trainer.step - data: {data.shape}")
        # print(data[:5,0])
        
        # Get $z$, $\mu$, and $\hat{\sigma}$
        z, mu, sigma_hat = self.encoder(data)

        # Concatenate $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$
        z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
        inputs = torch.cat([data[:-1], z_stack], 2)
        # Get mixture of distributions and $\hat{q}$
        dist, q_logits, _ = self.decoder(inputs, z, None)

        # $L_{KL}$
        kl_loss = self.kl_div_loss(sigma_hat, mu)
        if self.hp.use_eta:
            kl_loss *= self.eta_step

        # $L_R$
        reconstruction_loss = self.reconstruction_loss(mask, data[1:], dist, q_logits)
        # $Loss = L_R + w_{KL} L_{KL}$
        loss = reconstruction_loss + self.hp.kl_div_loss_weight * kl_loss

        # Only if we are in training state
        if is_training:
            # Set `grad` to zero
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            # Compute gradients
            loss.backward()
            # Clip gradients
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)
            # Optimize
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        return loss.item(), reconstruction_loss.item(), kl_loss.item(), batch_items

    def validate_one_epoch(self, epoch):
        total_items, total_loss, total_kl_loss, total_reconstruction_loss = 0, 0, 0, 0
        with torch.no_grad():    
            for batch in iter(self.valid_loader):
                loss, reconstruction_loss, kl_loss, batch_items = self.step(batch, is_training=False)

                total_loss += loss * batch_items
                total_reconstruction_loss += reconstruction_loss * batch_items
                total_kl_loss += kl_loss * batch_items
                total_items += batch_items
                
        avg_loss = total_loss / total_items
        avg_reconstruction_loss = total_reconstruction_loss / total_items
        avg_kl_loss = total_kl_loss / total_items
        self.log(dict(
            val_avg_loss=avg_loss,
            val_avg_reconstruction_loss=avg_reconstruction_loss,
            val_avg_kl_loss=avg_kl_loss,
            epoch=epoch))
        return avg_loss, avg_reconstruction_loss, avg_kl_loss

    def train_one_epoch(self, epoch, parent_progressbar=None):
        steps_per_epoch = len(self.train_loader)
        for idx, batch in enumerate(progress_bar(iter(self.train_loader), parent=parent_progressbar)):
            self.total_steps = idx + epoch * steps_per_epoch
            loss, reconstruction_loss, kl_loss, _ = self.step(batch, is_training=True)
            self.log(dict(
                loss=loss,
                reconstruction_loss=reconstruction_loss,
                kl_loss=kl_loss,
                epoch=epoch,
                learning_rate=self.learning_rate,
                eta_step=self.eta_step))
        # update learning rate, if use_lr_decay is enabled
        if self.hp.use_lr_decay:
            if self.learning_rate > self.hp.min_lr:
                self.learning_rate *= self.hp.lr_decay
            self.encoder_optimizer = self.update_lr(self.encoder_optimizer, self.learning_rate)
            self.decoder_optimizer = self.update_lr(self.decoder_optimizer, self.learning_rate)
        # update weight of KL loss, if use_eta is enabled
        if self.hp.use_eta:
            self.eta_step = 1-(1-self.hp.eta_min)*self.hp.eta_R

    def update_lr(self, optimizer, lr):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
        
    def train(self):
        mb = master_bar(range(self.hp.epochs))
        for epoch in mb:
            self.train_one_epoch(epoch=epoch, parent_progressbar=mb)
            val_avg_loss, *_ = self.validate_one_epoch(epoch)
            update_best_val = False
            if val_avg_loss < self.best_val_loss:
                self.best_val_loss = val_avg_loss
                update_best_val = True
                #if epoch % self.hp.save_every_n_epochs == 0:
                self.save(epoch=0)
                self.sample(epoch=0)
            mb.write(f"Finished epoch {epoch}. Validation Loss: {val_avg_loss}{' (new best)' if update_best_val else ''}")
