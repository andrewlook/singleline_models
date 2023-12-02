"""
---
title: Sketch RNN
summary: >
  This is an annotated PyTorch implementation of the Sketch RNN from paper A Neural Representation of Sketch Drawings.
  Sketch RNN is a sequence-to-sequence model that generates sketches of objects such as bicycles, cats, etc.
---

# Sketch RNN

This is an annotated [PyTorch](https://pytorch.org) implementation of the paper
[A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).

Sketch RNN is a sequence-to-sequence variational auto-encoder.
Both encoder and decoder are recurrent neural network models.
It learns to reconstruct stroke based simple drawings, by predicting
a series of strokes.
Decoder predicts each stroke as a mixture of Gaussian's.

### Getting data
Download data from [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset).
There is a link to download `npz` files in *Sketch-RNN QuickDraw Dataset* section of the readme.
Place the downloaded `npz` file(s) in `data/sketch` folder.
This code is configured to use `bicycle` dataset.
You can change this in configurations.

### Acknowledgements
- [PyTorch Sketch RNN](https://github.com/alexis-jacq/Pytorch-Sketch-RNN) project by [Alexis David Jacq](https://github.com/alexis-jacq)



### Improvements

- [ ] Log epoch and learning rate

- [ ] LR decay
- [ ] ETA decay (for KL loss)

- [ ] Dropout
- [ ] Layer Normalization
- [ ] Recurrent Dropout


"""


import io
import math
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from matplotlib import pyplot as plt
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset


class StrokesDataset(Dataset):
    """
    ## Dataset

    This class loads and pre-processes the data.
    """

    def __init__(self, dataset: np.array, max_seq_length: int, scale: Optional[float] = None):
        """
        `dataset` is a list of numpy arrays of shape [seq_len, 3].
        It is a sequence of strokes, and each stroke is represented by
        3 integers.
        First two are the displacements along x and y ($\Delta x$, $\Delta y$)
        and the last integer represents the state of the pen, $1$ if it's touching
        the paper and $0$ otherwise.
        """

        data = []
        # We iterate through each of the sequences and filter
        for seq in dataset:
            # Filter if the length of the sequence of strokes is within our range
            if 10 < len(seq) <= max_seq_length:
                # Clamp $\Delta x$, $\Delta y$ to $[-1000, 1000]$
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                # Convert to a floating point array and add to `data`
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)

        # We then calculate the scaling factor which is the
        # standard deviation of ($\Delta x$, $\Delta y$) combined.
        # Paper notes that the mean is not adjusted for simplicity,
        # since the mean is anyway close to $0$.
        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale

        # Get the longest sequence length among all sequences
        longest_seq_len = max([len(seq) for seq in data])

        # We initialize PyTorch data array with two extra steps for start-of-sequence (sos)
        # and end-of-sequence (eos).
        # Each step is a vector $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # Only one of $p_1, p_2, p_3$ is $1$ and the others are $0$.
        # They represent *pen down*, *pen up* and *end-of-sequence* in that order.
        # $p_1$ is $1$ if the pen touches the paper in the next step.
        # $p_2$ is $1$ if the pen doesn't touch the paper in the next step.
        # $p_3$ is $1$ if it is the end of the drawing.
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)
        # The mask array needs only one extra-step since it is for the outputs of the
        # decoder, which takes in `data[:-1]` and predicts next step.
        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # Scale and set $\Delta x, \Delta y$
            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / scale
            # $p_1$
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]
            # $p_2$
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]
            # $p_3$
            self.data[i, len_seq + 1:, 4] = 1
            # Mask is on until end of sequence
            self.mask[i, :len_seq + 1] = 1

        # Start-of-sequence is $(0, 0, 1, 0, 0)$
        self.data[:, 0, 2] = 1

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample"""
        return self.data[idx], self.mask[idx]


class BivariateGaussianMixture:
    """
    ## Bi-variate Gaussian mixture

    The mixture is represented by $\Pi$ and
    $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
    This class adjusts temperatures and creates the categorical and Gaussian
    distributions from the parameters.
    """

    def __init__(self, pi_logits: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor,
                 sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor):
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    @property
    def n_distributions(self):
        """Number of distributions in the mixture, $M$"""
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        """
        Adjust by temperature $\tau$
        """
        # $$\hat{\Pi_k} \leftarrow \frac{\hat{\Pi_k}}{\tau}$$
        self.pi_logits /= temperature
        # $$\sigma^2_x \leftarrow \sigma^2_x \tau$$
        self.sigma_x *= math.sqrt(temperature)
        # $$\sigma^2_y \leftarrow \sigma^2_y \tau$$
        self.sigma_y *= math.sqrt(temperature)

    def get_distribution(self):
        # Clamp $\sigma_x$, $\sigma_y$ and $\rho_{xy}$ to avoid getting `NaN`s
        sigma_x = torch.clamp_min(self.sigma_x, 1e-5)
        sigma_y = torch.clamp_min(self.sigma_y, 1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1 - 1e-5)

        # Get means
        mean = torch.stack([self.mu_x, self.mu_y], -1)
        # Get covariance matrix
        cov = torch.stack([
            sigma_x * sigma_x, rho_xy * sigma_x * sigma_y,
            rho_xy * sigma_x * sigma_y, sigma_y * sigma_y
        ], -1)
        cov = cov.view(*sigma_y.shape, 2, 2)

        # Create bi-variate normal distribution.
        #
        # 📝 It would be efficient to `scale_tril` matrix as `[[a, 0], [b, c]]`
        # where
        # $$a = \sigma_x, b = \rho_{xy} \sigma_y, c = \sigma_y \sqrt{1 - \rho^2_{xy}}$$.
        # But for simplicity we use co-variance matrix.
        # [This is a good resource](https://www2.stat.duke.edu/courses/Spring12/sta104.1/Lectures/Lec22.pdf)
        # if you want to read up more about bi-variate distributions, their co-variance matrix,
        # and probability density function.
        multi_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

        # Create categorical distribution $\Pi$ from logits
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)

        #
        return cat_dist, multi_dist


class EncoderRNN(nn.Module):
    """
    ## Encoder module

    This consists of a bidirectional LSTM
    """

    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        # Create a bidirectional LSTM taking a sequence of
        # $(\Delta x, \Delta y, p_1, p_2, p_3)$ as input.
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        # Head to get $\mu$
        self.mu_head = nn.Linear(2 * enc_hidden_size, d_z)
        # Head to get $\hat{\sigma}$
        self.sigma_head = nn.Linear(2 * enc_hidden_size, d_z)

    def forward(self, inputs: torch.Tensor, state=None):
        # The hidden state of the bidirectional LSTM is the concatenation of the
        # output of the last token in the forward direction and
        # first token in the reverse direction, which is what we want.
        # $$h_{\rightarrow} = encode_{\rightarrow}(S),
        # h_{\leftarrow} = encode←_{\leftarrow}(S_{reverse}),
        # h = [h_{\rightarrow}; h_{\leftarrow}]$$
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        # The state has shape `[2, batch_size, hidden_size]`,
        # where the first dimension is the direction.
        # We rearrange it to get $h = [h_{\rightarrow}; h_{\leftarrow}]$
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')

        # $\mu$
        mu = self.mu_head(hidden)
        # $\hat{\sigma}$
        sigma_hat = self.sigma_head(hidden)
        # $\sigma = \exp(\frac{\hat{\sigma}}{2})$
        sigma = torch.exp(sigma_hat / 2.)

        # Sample $z = \mu + \sigma \cdot \mathcal{N}(0, I)$
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))

        #
        return z, mu, sigma_hat


class DecoderRNN(nn.Module):
    """
    ## Decoder module

    This consists of a LSTM
    """

    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int):
        super().__init__()
        # LSTM takes $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ as input
        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)

        # Initial state of the LSTM is $[h_0; c_0] = \tanh(W_{z}z + b_z)$.
        # `init_state` is the linear transformation for this
        self.init_state = nn.Linear(d_z, 2 * dec_hidden_size)

        # This layer produces outputs for each of the `n_distributions`.
        # Each distribution needs six parameters
        # $(\hat{\Pi_i}, \mu_{x_i}, \mu_{y_i}, \hat{\sigma_{x_i}}, \hat{\sigma_{y_i}} \hat{\rho_{xy_i}})$
        self.mixtures = nn.Linear(dec_hidden_size, 6 * n_distributions)

        # This head is for the logits $(\hat{q_1}, \hat{q_2}, \hat{q_3})$
        self.q_head = nn.Linear(dec_hidden_size, 3)
        # This is to calculate $\log(q_k)$ where
        # $$q_k = \operatorname{softmax}(\hat{q})_k = \frac{\exp(\hat{q_k})}{\sum_{j = 1}^3 \exp(\hat{q_j})}$$
        self.q_log_softmax = nn.LogSoftmax(-1)

        # These parameters are stored for future reference
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size

    def forward(self, x: torch.Tensor, z: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # Calculate the initial state
        if state is None:
            # $[h_0; c_0] = \tanh(W_{z}z + b_z)$
            h, c = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            # `h` and `c` have shapes `[batch_size, lstm_size]`. We want to shape them
            # to `[1, batch_size, lstm_size]` because that's the shape used in LSTM.
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())

        # Run the LSTM
        outputs, state = self.lstm(x, state)

        # Get $\log(q)$
        q_logits = self.q_log_softmax(self.q_head(outputs))

        # Get $(\hat{\Pi_i}, \mu_{x,i}, \mu_{y,i}, \hat{\sigma_{x,i}},
        # \hat{\sigma_{y,i}} \hat{\rho_{xy,i}})$.
        # `torch.split` splits the output into 6 tensors of size `self.n_distribution`
        # across dimension `2`.
        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = \
            torch.split(self.mixtures(outputs), self.n_distributions, 2)

        # Create a bi-variate Gaussian mixture
        # $\Pi$ and 
        # $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        # where
        # $$\sigma_{x,i} = \exp(\hat{\sigma_{x,i}}), \sigma_{y,i} = \exp(\hat{\sigma_{y,i}}),
        # \rho_{xy,i} = \tanh(\hat{\rho_{xy,i}})$$
        # and
        # $$\Pi_i = \operatorname{softmax}(\hat{\Pi})_i = \frac{\exp(\hat{\Pi_i})}{\sum_{j = 1}^3 \exp(\hat{\Pi_j})}$$
        #
        # $\Pi$ is the categorical probabilities of choosing the distribution out of the mixture
        # $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
        dist = BivariateGaussianMixture(pi_logits, mu_x, mu_y,
                                        torch.exp(sigma_x), torch.exp(sigma_y), torch.tanh(rho_xy))

        return dist, q_logits, state


class ReconstructionLoss(nn.Module):
    """
    ## Reconstruction Loss
    """

    def forward(self, mask: torch.Tensor, target: torch.Tensor,
                 dist: 'BivariateGaussianMixture', q_logits: torch.Tensor):
        # Get $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        pi, mix = dist.get_distribution()
        # `target` has shape `[seq_len, batch_size, 5]` where the last dimension is the features
        # $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # We want to get $\Delta x, \Delta$ y and get the probabilities from each of the distributions
        # in the mixture $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
        #
        # `xy` will have shape `[seq_len, batch_size, n_distributions, 2]`
        xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)
        # Calculate the probabilities
        # $$p(\Delta x, \Delta y) =
        # \sum_{j=1}^M \Pi_j \mathcal{N} \big( \Delta x, \Delta y \vert
        # \mu_{x,j}, \mu_{y,j}, \sigma_{x,j}, \sigma_{y,j}, \rho_{xy,j}
        # \big)$$
        probs = torch.sum(pi.probs * torch.exp(mix.log_prob(xy)), 2)

        # $$L_s = - \frac{1}{N_{max}} \sum_{i=1}^{N_s} \log \big (p(\Delta x, \Delta y) \big)$$
        # Although `probs` has $N_{max}$ (`longest_seq_len`) elements, the sum is only taken
        # upto $N_s$ because the rest is masked out.
        #
        # It might feel like we should be taking the sum and dividing by $N_s$ and not $N_{max}$,
        # but this will give higher weight for individual predictions in shorter sequences.
        # We give equal weight to each prediction $p(\Delta x, \Delta y)$ when we divide by $N_{max}$
        loss_stroke = -torch.mean(mask * torch.log(1e-5 + probs))

        # $$L_p = - \frac{1}{N_{max}} \sum_{i=1}^{N_{max}} \sum_{k=1}^{3} p_{k,i} \log(q_{k,i})$$
        loss_pen = -torch.mean(target[:, :, 2:] * q_logits)

        # $$L_R = L_s + L_p$$
        return loss_stroke + loss_pen


class KLDivLoss(nn.Module):
    """
    ## KL-Divergence loss

    This calculates the KL divergence between a given normal distribution and $\mathcal{N}(0, 1)$
    """

    def forward(self, sigma_hat: torch.Tensor, mu: torch.Tensor):
        # $$L_{KL} = - \frac{1}{2 N_z} \bigg( 1 + \hat{\sigma} - \mu^2 - \exp(\hat{\sigma}) \bigg)$$
        return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat))


class Sampler:
    """
    ## Sampler

    This samples a sketch from the decoder and plots it
    """

    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        self.decoder = decoder
        self.encoder = encoder

    def sample(self, data: torch.Tensor, temperature: float, fpath: str):
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

        # Plot the sequence of strokes
        self.plot(seq, fpath)

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

    @staticmethod
    def plot(_seq: torch.Tensor, fpath: str):
        seq = torch.zeros_like(_seq)
        # Take the cumulative sums of $(\Delta x, \Delta y)$ to get $(x, y)$
        seq[:, 0:2] = torch.cumsum(_seq[:, 0:2], dim=0)
        # Create a new numpy array of the form $(x, y, q_2)$
        seq[:, 2] = _seq[:, 3]
        seq = seq[:, 0:3].detach().cpu().numpy()

        # Split the array at points where $q_2$ is $1$.
        # i.e. split the array of strokes at the points where the pen is lifted from the paper.
        # This gives a list of sequence of strokes.
        strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
        # Plot each sequence of strokes
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        # Don't show axes
        plt.axis('off')

        # # Show the plot
        # plt.show()

        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            img = Image.open(buf)
            img.save(fpath)
            buf.seek(0)
            buf.truncate()
        return img


class HParams():
    architecture = 'Pytorch-LabML'
    dataset_name: str = 'look_i16__minn10_epsilon1'
    epochs = 50000

    learning_rate = 1e-3
    
    # Encoder and decoder sizes
    enc_hidden_size = 256
    dec_hidden_size = 512

    # Batch size
    batch_size = 100

    # Number of features in $z$
    d_z = 128
    # Number of distributions in the mixture, $M$
    n_distributions = 20

    # Weight of KL divergence loss, $w_{KL}$
    kl_div_loss_weight = 0.5
    # Gradient clipping
    grad_clip = 1.
    # Temperature $\tau$ for sampling
    temperature = 0.4

    # Filter out stroke sequences longer than $200$
    max_seq_length = 200


class Trainer():
    # Device configurations to pick the device to run the experiment
    device: str
    
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

    def __init__(self, hp: HParams, device="cuda", use_wandb=False, models_dir="models"):
        self.hp = hp
        config = {k: getattr(hp, k) for k in hp.__dir__() if not k.startswith('__')}
        print(config)

        self.device = device
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            run = wandb.init(
                project='sketchrnn-pytorch',
                entity='andrewlook',
                config=config,
            )
            self.run_id = run.id
        else:
            self.run_id = f"{math.floor(np.random.rand() * 1e6):07d}"

        self.models_dir = Path(models_dir)
        self.run_dir = self.models_dir / self.run_id
        if not os.path.isdir(self.run_dir):
            os.makedirs(self.run_dir)
        
        # Initialize encoder & decoder
        self.encoder = EncoderRNN(self.hp.d_z, self.hp.enc_hidden_size).to(self.device)
        self.decoder = DecoderRNN(self.hp.d_z, self.hp.dec_hidden_size, self.hp.n_distributions).to(self.device)
        if self.use_wandb:
            wandb.watch((self.encoder, self.decoder), log="all", log_freq=10, log_graph=True)

        self.learning_rate = self.hp.learning_rate
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), self.learning_rate)

        # Create sampler
        self.sampler = Sampler(self.encoder, self.decoder)

        # `npz` file path is `data/quickdraw/[DATASET NAME].npz`
        base_path = Path("data/quickdraw")
        path = base_path / f'{self.hp.dataset_name}.npz'
        # Load the numpy file
        dataset = np.load(str(path), encoding='latin1', allow_pickle=True)

        # Create training dataset
        self.train_dataset = StrokesDataset(dataset['train'], self.hp.max_seq_length)
        # Create validation dataset
        self.valid_dataset = StrokesDataset(dataset['valid'], self.hp.max_seq_length, self.train_dataset.scale)

        # Create training data loader
        self.train_loader = DataLoader(self.train_dataset, self.hp.batch_size, shuffle=True)
        # Create validation data loader
        self.valid_loader = DataLoader(self.valid_dataset, self.hp.batch_size)

        self.total_steps = 0
        self.valid_idxs = [np.random.choice(len(self.valid_dataset)) for _ in range(5)]

    def step(self, batch: Any, is_training=False):
        self.encoder.train(is_training)
        self.decoder.train(is_training)

        # Move `data` and `mask` to device and swap the sequence and batch dimensions.
        # `data` will have shape `[seq_len, batch_size, 5]` and
        # `mask` will have shape `[seq_len, batch_size]`.
        data = batch[0].to(self.device).transpose(0, 1)
        mask = batch[1].to(self.device).transpose(0, 1)
        batch_items = len(data)
        
        # Get $z$, $\mu$, and $\hat{\sigma}$
        z, mu, sigma_hat = self.encoder(data)

        # Concatenate $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$
        z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
        inputs = torch.cat([data[:-1], z_stack], 2)
        # Get mixture of distributions and $\hat{q}$
        dist, q_logits, _ = self.decoder(inputs, z, None)

        # $L_{KL}$
        kl_loss = self.kl_div_loss(sigma_hat, mu)
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
        return loss, reconstruction_loss, kl_loss, batch_items

    def validate_one_epoch(self, epoch):
        total_items = 0
        total_loss = 0
        total_kl_loss = 0
        total_reconstruction_loss = 0

        with torch.no_grad():    
            for idx, batch in enumerate(iter(self.valid_loader)):
                loss, reconstruction_loss, kl_loss, batch_items = self.step(batch, is_training=False)

                total_items += batch_items
                total_loss += loss.item() * batch_items
                total_reconstruction_loss += reconstruction_loss.item() * batch_items
                total_kl_loss += kl_loss.item() * batch_items

        avg_loss = total_loss / total_items
        avg_reconstruction_loss = total_reconstruction_loss / total_items
        avg_kl_loss = total_kl_loss / total_items

        if self.use_wandb:
            validation_losses = dict(
                val_avg_loss=avg_loss,
                val_avg_reconstruction_loss=avg_reconstruction_loss,
                val_avg_kl_loss=avg_kl_loss,
                epoch=epoch,
            )
            wandb.log(validation_losses, step=self.total_steps)

        return avg_loss, avg_reconstruction_loss, avg_kl_loss

    def train_one_epoch(self, epoch, parent_progressbar=None):
        steps_per_epoch = len(self.train_loader)
        for idx, batch in enumerate(progress_bar(iter(self.train_loader), parent=parent_progressbar)):
            step_num = idx + epoch * steps_per_epoch
            self.total_steps = step_num
            loss, reconstruction_loss, kl_loss, batch_items = self.step(batch, is_training=True)
            if self.use_wandb:
                log_values = dict(
                    loss=loss,
                    reconstruction_loss=reconstruction_loss,
                    kl_loss=kl_loss,
                    learning_rate=self.learning_rate,
                    epoch=epoch,
                )
                wandb.log(log_values, step=step_num)

    def train(self):
        validate_every_n_epochs = 2
        save_every_n_epochs = 100

        mb = master_bar(range(self.hp.epochs))
        for epoch in mb:
            self.train_one_epoch(epoch=epoch, parent_progressbar=mb)
            mb.write(f'Finished epoch {epoch}.')
            if epoch % validate_every_n_epochs == 0:
                self.validate_one_epoch(epoch)
            if epoch % save_every_n_epochs == 0:
                self.save(epoch)
                self.sample(epoch)

    def sample(self, epoch, display=False):
        orig_paths = []
        decoded_paths = []
        for idx in self.valid_idxs:
            orig_path = self.run_dir / f'sample_runid-{self.run_id}_{idx:04d}_epoch_{epoch:05d}_orig.png'
            decoded_path = self.run_dir / f'sample_runid-{self.run_id}_{idx:04d}_epoch_{epoch:05d}_decoded.png'

            # Randomly pick a sample from validation dataset to encoder
            data, *_ = self.valid_dataset[idx]
            self.sampler.plot(data, orig_path)

            # Add batch dimension and move it to device
            data_batched = data.unsqueeze(1).to(self.device)
            # Sample
            self.sampler.sample(data_batched, self.hp.temperature, decoded_path)

            if display:
                Image.open(orig_path).show()
                Image.open(decoded_path).show()
            orig_paths.append(orig_path)
            decoded_paths.append(decoded_path)
        return sorted(orig_paths), sorted(decoded_paths)
    
    def save(self, epoch):
        torch.save(self.encoder.state_dict(), \
            Path(self.run_dir) / f'encoderRNN_runid-{self.run_id}_epoch_{epoch:05d}.pth')
        torch.save(self.decoder.state_dict(), \
            Path(self.run_dir) / f'decoderRNN_runid-{self.run_id}epoch_{epoch:05d}.pth')

    def load(self, epoch):
        saved_encoder = torch.load(Path(self.run_dir) / f'encoderRNN_runid-{self.run_id}_epoch_{epoch:05d}.pth')
        saved_decoder = torch.load(Path(self.run_dir) / f'decoderRNN_runid-{self.run_id}_epoch_{epoch:05d}.pth')
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)


def main():
    use_wandb = False

    hp = HParams()
    hp.learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(hp=hp,
                      device=device,
                      use_wandb=use_wandb)
    trainer.train()
        

if __name__ == "__main__":
    main()
