import numpy as np
import torch
from sklearn.model_selection import train_test_split


def get_X(N=1_000, I=50, Q=30, N_test=200):
    N = N + N_test

    X = torch.randn([N * I, Q], dtype=torch.float32)

    X = (X - X.mean(0)) / X.std(0)
    X = X / np.sqrt(X.shape[1])
    X = X.reshape([N, I, Q])

    return X


def simulate(X, v_beta=0.5, v_gamma=0.8, b=-1, F=None, P=1):
    if F is None:
        F = torch.ones([X.shape[0], 1])

    # simulate single phenotype
    K = F.shape[1]
    b = b * torch.ones([K, P])
    v_beta = v_beta * torch.ones(P)
    v_gamma = v_gamma * torch.ones(P)

    # sample weights
    gamma = torch.randn((X.shape[2], v_gamma.shape[0]))
    _w = torch.einsum("nik,kp->nip", X, gamma)
    _scale_w = torch.sqrt(v_gamma / _w.var([0, 1]))
    gamma = _scale_w[None, :] * gamma
    _w = _scale_w[None, None, :] * _w

    w = torch.softmax(_w, dim=1)

    # sample z
    beta = torch.randn((X.shape[2], v_beta.shape[0]))
    beta = beta / torch.sqrt((beta**2).mean(0, keepdim=True))
    z = torch.einsum("nik,kp->nip", X, beta)
    u = torch.einsum("nip,nip->np", w, z)
    u = (u - u.mean(0)) / u.std(0)
    u = torch.sqrt(v_beta) * u
    beta = torch.sqrt(v_beta) * beta

    # compute rates
    logits = F.mm(b) + u

    # sample Y
    Y = torch.distributions.Binomial(2, logits=logits).sample()

    return F, Y, u, w


def split_data(Xs, test_size=200, val_size=0.0, test_rs=127, val_rs=412):
    idxs_all = np.arange(Xs[0].shape[0])
    idxs = {}
    idxs["train_val"], idxs["test"] = train_test_split(idxs_all, test_size=test_size, random_state=test_rs)

    if not np.isclose(val_size, 0):
        idxs["train"], idxs["val"] = train_test_split(idxs["train_val"], test_size=val_size, random_state=val_rs)
    else:
        idxs["train"] = idxs["train_val"]
    del idxs["train_val"]

    outs = []
    for X in Xs:
        out = {}
        for key in idxs.keys():
            out[key] = X[idxs[key]]
        outs.append(out)
    return outs


def load_simulation(seed=42, N=1_000, I=50, Q=30, N_test=200, P=1, v_beta=0.5, v_gamma=0.8, b=-1, F=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = get_X(N, I, Q, N_test)
    data = simulate(X, v_beta, v_gamma, b, F, P)
    X, F, Y, u, w = split_data((X, *data))
    return X, F, Y, u, w
