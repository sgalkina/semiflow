from math import prod

import torch
from torch import nn
import numpy as np

from multivae.data.datasets import MnistSvhn
from multivae.data.datasets import IncompleteDataset, DatasetOutput
import torch
import pickle

# Dataset
DATA_PATH = './MNIST-SVHN'
train_set = MnistSvhn(data_path = DATA_PATH, split="train", data_multiplication=1, download=True)
test_set = MnistSvhn(data_path = DATA_PATH, split="test", data_multiplication=1, download=True)

L = len(train_set)

# Define random data samples
data = dict(
    mnist = torch.stack([b['data']['mnist'] for b in train_set]),
    svhn = torch.stack([b['data']['svhn'] for b in train_set])
)
# Define random masks : masks are boolean tensors: True indicates the modality is available. 

# Arbitrary labels (optional)
labels = torch.stack([b['labels'] for b in train_set]).tolist()

N = 1

def make_balanced_mask(labels):
    classes = list(set(labels))
    positions = {k: [] for k in classes}
    for i, l in enumerate(labels):
        positions[l].append(i)
    print({k: len(v) for k, v in positions.items()})
    final_positions = {}
    for k, v in positions.items():
        masked = torch.bernoulli(N*torch.ones((len(v),))).bool().tolist()
        for i, j in zip(v, masked):
            final_positions[i] = j
    mask = [final_positions[i] for i in range(len(labels))]
    print(sum(mask) / len(mask))
    return mask

masks = dict(
    mnist = [1]*len(labels),
    svhn = make_balanced_mask(labels),
)


DIR = './datasets_incomplete/'

def save_dataset():
    with open(DIR + f'{N}_masks_full.pickle', 'wb') as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(DIR + f'{N}_labels_full.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(DIR + f'{N}_data_full.npz', mnist=data['mnist'].numpy(), svhn=data['svhn'].numpy())

save_dataset()