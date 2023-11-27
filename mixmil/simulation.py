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


def simulate(X, v_z=0.5, v_u=0.8, b=-1, F=None, P=1):
    if F is None:
        F = torch.ones([X.shape[0], 1])

    # simulate single phenotype
    b = b * torch.ones([1, P])
    v_z = v_z * torch.ones(P)
    v_u = v_u * torch.ones(P)

    # sample u
    beta_u = torch.randn((X.shape[2], v_u.shape[0]))
    u = torch.einsum("nik,kp->nip", X, beta_u)
    _scale_u = torch.sqrt(v_u / u.var([0, 1]))
    beta_u = _scale_u[None, :] * beta_u
    u = _scale_u[None, None, :] * u

    w = torch.softmax(u, dim=1)

    # sample z
    beta_z = torch.randn((X.shape[2], v_z.shape[0]))
    beta_z = beta_z / torch.sqrt((beta_z**2).mean())
    x = torch.einsum("nik,kp->nip", X, beta_z)
    z = torch.einsum("nip,nip->np", w, x)
    z = (z - z.mean(0)) / z.std(0)
    z = torch.sqrt(v_z) * z
    beta_z = torch.sqrt(v_z) * beta_z

    # compute rates
    logits = F.mm(b) + z
    probs = torch.sigmoid(logits)

    # sample Y
    Y = torch.distributions.Binomial(2, logits=logits).sample().data.float()

    more = {
        "u": u,
        "w": w,
        "x": x,
        "xmil": z,
        "logits": logits,
        "probs": probs,
        "beta_u": beta_u,
        "beta_z": beta_z,
    }

    return Y, F, more


def split_data(Xs, test_size=200, val_size=0.1, test_rs=127, val_rs=412):
    # define indices
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


def load_simulation(sim_seed=0):
    np.random.seed(sim_seed)
    torch.manual_seed(sim_seed)

    X = get_X()
    Ya, Fa, more = simulate(X)
    data = [Ya, X, Fa, more["xmil"], more["w"]]
    Y, X, F, xmil, w = split_data(data)
    more = dict(xmil=xmil, w=w)
    return X, Y, F, more
