import io

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

# from .model import BivariateGaussianMixture, DecoderRNN, EncoderRNN


class Sampler:
    """
    This samples a sketch from the decoder and plots it
    """

    # def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
    #     self.decoder = decoder
    #     self.encoder = encoder

    # def sample(self, data: torch.Tensor, temperature: float, fpath: str = None):
    #     # $N_{max}$
    #     longest_seq_len = len(data)

    #     # Get $z$ from the encoder
    #     z, _, _ = self.encoder(data)

    #     # Start-of-sequence stroke is $(0, 0, 1, 0, 0)$
    #     s = data.new_tensor([0, 0, 1, 0, 0])
    #     seq = [s]
    #     # Initial decoder is `None`.
    #     # The decoder will initialize it to $[h_0; c_0] = \tanh(W_{z}z + b_z)$
    #     state = None

    #     # We don't need gradients
    #     with torch.no_grad():
    #         # Sample $N_{max}$ strokes
    #         for i in range(longest_seq_len):
    #             # $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ is the input to the decoder
    #             data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
    #             # Get $\Pi$, $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$,
    #             # $q$ and the next state from the decoder
    #             dist, q_logits, state = self.decoder(data, z, state)
    #             # Sample a stroke
    #             s = self._sample_step(dist, q_logits, temperature)
    #             # Add the new stroke to the sequence of strokes
    #             seq.append(s)
    #             # Stop sampling if $p_3 = 1$. This indicates that sketching has stopped
    #             if s[4] == 1:
    #                 break

    #     # Create a PyTorch tensor of the sequence of strokes
    #     seq = torch.stack(seq)

    #     # Plot the sequence of strokes
    #     self.plot(seq, fpath=fpath)

    # @staticmethod
    # def _sample_step(dist: BivariateGaussianMixture, q_logits: torch.Tensor, temperature: float):
    #     # Set temperature $\tau$ for sampling. This is implemented in class `BivariateGaussianMixture`.
    #     dist.set_temperature(temperature)
    #     # Get temperature adjusted $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
    #     pi, mix = dist.get_distribution()
    #     # Sample from $\Pi$ the index of the distribution to use from the mixture
    #     idx = pi.sample()[0, 0]

    #     # Create categorical distribution $q$ with log-probabilities `q_logits` or $\hat{q}$
    #     q = torch.distributions.Categorical(logits=q_logits / temperature)
    #     # Sample from $q$
    #     q_idx = q.sample()[0, 0]

    #     # Sample from the normal distributions in the mixture and pick the one indexed by `idx`
    #     xy = mix.sample()[0, 0, idx]

    #     # Create an empty stroke $(\Delta x, \Delta y, q_1, q_2, q_3)$
    #     stroke = q_logits.new_zeros(5)
    #     # Set $\Delta x, \Delta y$
    #     stroke[:2] = xy
    #     # Set $q_1, q_2, q_3$
    #     stroke[q_idx + 2] = 1
    #     #
    #     return stroke

    @staticmethod
    def plot(_seq: torch.Tensor, fpath = None, figsize=(6,6)):
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

        fig = plt.figure(figsize=figsize)

        # Plot each sequence of strokes
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1], figure=fig)
        # Don't show axes
        plt.axis('off')

        if not fpath:
            plt.show()
            return
        
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            img = Image.open(buf)
            img.save(fpath)
            buf.seek(0)
            buf.truncate()
            return img