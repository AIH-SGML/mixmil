import numpy as np
import scipy.linalg as la
import statsmodels.api as sm
import torch
from sklearn.linear_model import LogisticRegressionCV
from tqdm.auto import trange

from mixmil.data import xgower_factor


def regressOut(Y, X, return_b=False):
    """
    regresses out X from Y
    """
    Xd = la.pinv(X)
    b = Xd.dot(Y)
    Y_out = Y - X.dot(b)
    if return_b:
        return Y_out, b
    else:
        return Y_out


def get_binomial_init_params(X, F, y):
    ident = np.zeros((X.shape[1]), dtype=int)
    model = sm.BinomialBayesMixedGLM(y, F, X, ident).fit_vb()

    xmil_train = np.dot(X, model.vc_mean)[::2]

    _scale = xmil_train.std() / np.sqrt((model.vc_mean**2).mean(0))
    mu_z = _scale * model.vc_mean
    sd_z = _scale * model.vc_sd
    var_z = (mu_z**2 + sd_z**2).mean().reshape(1)
    alpha = model.fe_mean

    return mu_z, sd_z, var_z, alpha


def get_lr_init_params(X, Y):
    model = LogisticRegressionCV(
        Cs=10,
        fit_intercept=True,
        penalty="l2",
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=8,
        verbose=0,
        random_state=42,
        max_iter=1000,
        refit=True,
    )

    model.fit(X, Y.ravel())

    alpha = model.intercept_[None]
    beta_z = model.coef_.T

    # Compute Xmil and reparametrize
    Xmil = X.dot(beta_z)
    xm = Xmil.mean(0)[None]
    xs = Xmil.std(0)[None]
    alpha = alpha + xm
    mu_z = xs * beta_z / np.sqrt((beta_z**2).mean(0)[None])
    sd_z = np.sqrt(0.1 * ((mu_z) ** 2).mean()) * np.ones_like(mu_z)

    # init prior
    var_z = (mu_z**2 + sd_z**2).mean().reshape(1, 1)

    return [torch.Tensor(el) for el in (mu_z, sd_z, var_z, alpha)]


def _list2tensor(listo):
    return torch.Tensor(np.stack(listo, axis=1))


def get_init_params(Xs, F, Y, likelihood, n_trials):
    if likelihood == "categorical":
        idx = np.concatenate([np.full(x.shape[0], i) for i, x in enumerate(Xs)], axis=0)
        Fr = np.concatenate([F[[i]] for i in idx], axis=0)
        Xr = np.concatenate(Xs, axis=0) if isinstance(Xs, list) else Xs.reshape(-1, Xs.shape[-1])
        Xr = regressOut(Xr, Fr)
        Xs = [Xr[idx == i] for i in range(idx.max() + 1)]

    Xm = np.concatenate([x.mean(0, keepdims=True).numpy() for x in Xs], axis=0)
    Xm = (Xm - Xm.mean(0, keepdims=True)) / xgower_factor(Xm)
    Fe, Ye = F.numpy(), Y.long().numpy()

    if likelihood == "binomial" and n_trials == 2:
        Xm = Xm.repeat(2, axis=0)
        Fe = Fe.repeat(2, axis=0)
        to_expanded = np.array(([[0, 0], [1, 0], [1, 1]]))
        Ye = to_expanded[Y.long().numpy().T].transpose(1, 2, 0).reshape(-1, Y.shape[1])

    if likelihood == "binomial":
        results = [get_binomial_init_params(Xm, Fe, Ye[:, p]) for p in trange(Ye.shape[1], desc="GLMM Init")]
        results = [_list2tensor(listo) for listo in zip(*results)]

    else:
        results = get_lr_init_params(Xm, Ye)

    mu_z, sd_z, var_z, alpha = results

    return mu_z, sd_z, var_z, alpha
