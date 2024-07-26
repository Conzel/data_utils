from torchvision.datasets import CIFAR10
import torchvision
from pathlib import Path
from data_utils.datasets import shuffled
from torch.utils.data import DataLoader, TensorDataset
import torch


def test_same_values():
    vals = torch.arange(0, 1000, 1)
    ds = TensorDataset(vals)
    train_dataloader_orig = DataLoader(ds, batch_size=32, shuffle=False)
    train_dataloader_shuff = DataLoader(shuffled(ds), batch_size=32, shuffle=False)

    samples_orig = []
    samples_shuff = []
    for xs in train_dataloader_orig:
        for x in xs[0]:
            samples_orig.append(x)
    for xs in train_dataloader_shuff:
        for x in xs[0]:
            samples_shuff.append(x)

    assert torch.allclose(torch.tensor(samples_orig), vals)
    assert torch.allclose(torch.tensor(samples_shuff).sort()[0], vals)


def test_is_shuffled():
    vals = torch.arange(0, 1000, 1)
    ds = TensorDataset(vals)
    train_dataloader_orig = DataLoader(ds, batch_size=128, shuffle=False)
    train_dataloader_shuff = DataLoader(shuffled(ds), batch_size=128, shuffle=False)

    samples_orig = []
    samples_shuff = []
    for xs in train_dataloader_orig:
        for x in xs[0]:
            samples_orig.append(x)
    for xs in train_dataloader_shuff:
        for x in xs[0]:
            samples_shuff.append(x)

    assert torch.allclose(torch.tensor(samples_orig), vals)
    assert not torch.allclose(torch.tensor(samples_shuff), vals)
