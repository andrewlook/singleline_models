# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/sketch_rnn/03_sampler.ipynb.

# %% auto 0
__all__ = ['Sampler']

# %% ../../nbs/sketch_rnn/03_sampler.ipynb 4
import io

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image


# %% ../../nbs/sketch_rnn/03_sampler.ipynb 5
class Sampler:
    """
    This samples a sketch from the decoder and plots it
    """

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
