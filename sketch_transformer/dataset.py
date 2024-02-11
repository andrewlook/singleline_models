
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def random_scale(data, random_scale_factor=0.15):
    """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
    x_scale_factor = (
        np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
    y_scale_factor = (
        np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
    result = np.copy(data)
    result[:, 0] *= x_scale_factor
    result[:, 1] *= y_scale_factor
    return result


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


import faiss
import json

class Stroke3Tokenizer(object):

    def __init__(self, centroids_zero, centroids_one):
        self.centroids_zero = centroids_zero
        print(self.centroids_zero.shape)
        self.kmeans_zero = faiss.IndexFlatL2(self.centroids_zero.shape[1])
        self.kmeans_zero.add(self.centroids_zero)

        self.centroids_one = centroids_one
        self.kmeans_one = faiss.IndexFlatL2(self.centroids_one.shape[1])
        self.kmeans_one.add(self.centroids_one)

    def save(self, base_dir, extra_suffix=""):
        CENTROIDS_FNAME = base_dir / f"stroke3_centroids{extra_suffix}.json"
        with open(CENTROIDS_FNAME, 'w') as outfile:
            payload = {
                'centroids_zero': self.centroids_zero.tolist(),
                'centroids_one': self.centroids_one.tolist(),
            }
            json.dump(payload, outfile, indent=2)
        print(f"wrote {CENTROIDS_FNAME}")

    @staticmethod
    def load(base_dir, extra_suffix=""):
        CENTROIDS_FNAME = base_dir / f"stroke3_centroids{extra_suffix}.json"
        with open(CENTROIDS_FNAME, 'r') as infile:
            payload = json.load(infile)
        centroids_zero = np.array(payload['centroids_zero'], dtype=np.float32)
        centroids_one = np.array(payload['centroids_one'], dtype=np.float32)

        tokenizer = Stroke3Tokenizer(centroids_zero, centroids_one)
        return tokenizer

    def encode(self, input_deltas):
        """
        total vocabulary size is len(centroids_zero) + len(centroids_one).

        for a point with lift_pen=0, the "word" index in the vocabulary
        is equal to its position within the centroids_zero list.

        for a point with lift_pen=1, the "word" index in the vocabulary
        is equal to its position within the centroids_one list, PLUS
        the total length of the centroids_zero list.
        """
        D0, I0 = self.kmeans_zero.search(input_deltas, 1)
        D1, I1 = self.kmeans_one.search(input_deltas, 1)
        
        tokens = []
        for idx in range(input_deltas.shape[0]):
            row = input_deltas[idx]
            if row[2] == 0:
                tokens.append(I0[idx][0])
            elif row[2] == 1:
                tokens.append(I1[idx][0] + self.centroids_zero.shape[0])
            else:
                raise Exception ('didnt find a 0 or 1 in the lift_pen column')
        return tokens

    def decode(self, tokens):
        num_zero = self.centroids_zero.shape[0]
        num_one = self.centroids_one.shape[0]
        decoded = []
        for tok in tokens:
            if tok < 0:
                raise Exception("invalid token index")
            if tok < num_zero:
                decoded.append(self.centroids_zero[tok])
            elif tok < (num_zero + num_one):
                decoded.append(self.centroids_one[tok - num_zero])
            else:
                raise Exception("invalid token index")
        return np.array(decoded)


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
        for idx, seq in enumerate(dataset):
            if len(seq) < 10:
                print(f"filtering out {idx} - length: {len(seq)}")
                continue
            elif len(seq) > max_seq_length:
                print(f"truncating {idx} - length: {len(seq)}")
                seq = seq[:max_seq_length]
            # Clamp $\Delta x$, $\Delta y$ to $[-1000, 1000]$
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            # Convert to a floating point array and add to `data`
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
        print(f"finished filtering - len(dataset) = {len(dataset)}, len(data) = {len(data)}")

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
