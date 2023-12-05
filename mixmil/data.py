import numpy as np
import torch
from torch.utils.data import Dataset

from mixmil.simulation import load_simulation


def setup_scatter(Xs):
    device = Xs[0].device
    x = torch.cat(Xs, dim=0)
    i = torch.cat([torch.full((x.shape[0],), idx) for idx, x in enumerate(Xs)]).to(device)
    i_ptr = torch.cat([torch.tensor([0], device=device), i.bincount().cumsum(0)])
    return x, i, i_ptr


def xgower_factor(X):
    a = np.power(X, 2).sum()
    b = X.dot(X.sum(0)).sum()
    return np.sqrt((a - b / X.shape[0]) / (X.shape[0] - 1))


class MILDataset(Dataset):
    def __init__(self, Xs, F, Y):
        self.Xs = Xs
        self.F = F
        self.Y = Y

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        X = self.Xs[idx]
        F = self.F[idx]
        Y = self.Y[idx]
        return X, F, Y


def mil_collate_fn(batch):
    X = [item[0] for item in batch]
    F = torch.stack([item[1] for item in batch])
    Y = torch.stack([item[2] for item in batch])
    return X, F, Y


def normalize_feats(X, norm_factor="std_sqrt"):
    assert norm_factor in ["std", "std_sqrt"]
    train_data = (
        torch.cat(X["train"], dim=0) if isinstance(X["train"], list) else X["train"].reshape(-1, X["train"].shape[2])
    )
    mean = train_data.mean(0, keepdims=True)
    std = train_data.std(0, keepdims=True)
    factor = std * np.sqrt(train_data.shape[1]) if norm_factor == "std_sqrt" else std

    for key in X:
        X[key] = [(x - mean) / factor for x in X[key]] if isinstance(X[key], list) else (X[key] - mean) / factor

    return X


def load_data(dataset="simulation", norm_factor="std_sqrt", **kwargs):
    if dataset == "simulation":
        X, F, Y, u, w = load_simulation(**kwargs)
        X = normalize_feats(X, norm_factor)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return X, F, Y, u, w
