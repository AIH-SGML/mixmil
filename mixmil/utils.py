import numpy as np
import scipy.linalg as la
import statsmodels.api as sm
import torch
from sklearn.linear_model import LogisticRegressionCV
from tqdm.auto import trange

from mixmil.data import xgower_factor


def regressOut(Y, X, return_b=False, return_pinv=False):
    """
    regresses out X from Y
    """
    Xd = la.pinv(X)
    b = Xd.dot(Y)
    Y_out = Y - X.dot(b)
    out = [Y_out]
    if return_b:
        out.append(b)
    if return_pinv:
        out.append(Xd)
    return out if len(out) > 1 else out[0]


def _get_single_binomial_init_params(X, F, y):
    ident = np.zeros((X.shape[1]), dtype=int)
    model = sm.BinomialBayesMixedGLM(y, F, X, ident).fit_vb()

    u = np.dot(X, model.vc_mean)[::2]

    _scale = u.std() / np.sqrt((model.vc_mean**2).mean(0))
    mu_beta = _scale * model.vc_mean
    sd_beta = _scale * model.vc_sd
    var_z = (mu_beta**2 + sd_beta**2).mean().reshape(1)
    alpha = model.fe_mean

    return mu_beta, sd_beta, var_z, alpha


def get_binomial_init_params(X, F, Y):
    results = [_get_single_binomial_init_params(X, F, Y[:, p]) for p in trange(Y.shape[1], desc="GLMM Init")]

    mu_beta, sd_beta, var_z, alpha = [_list2tensor(listo) for listo in zip(*results)]

    return mu_beta, sd_beta, var_z, alpha


def get_lr_init_params(X, Y, b, Fiv):
    model = LogisticRegressionCV(
        Cs=10,
        fit_intercept=True,
        penalty="l2",
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=1,
        verbose=0,
        random_state=42,
        max_iter=1000,
        refit=True,
    )

    model.fit(X, Y.ravel())

    alpha = model.intercept_[None]
    beta = model.coef_.T

    # Compute bag prediction u and reparametrize
    u = X.dot(beta)
    um = u.mean(0)[None]
    us = u.std(0)[None]
    alpha = alpha + um
    mu_beta = us * beta / np.sqrt((beta**2).mean(0)[None])
    sd_beta = np.sqrt(0.1 * (mu_beta**2).mean()) * np.ones_like(mu_beta)

    alpha = Fiv.dot(np.ones((Fiv.shape[1], 1))).dot(alpha) - b.dot(mu_beta)

    # init prior
    var_z = (mu_beta**2 + sd_beta**2).mean(axis=0, keepdims=True)

    return [torch.Tensor(el) for el in (mu_beta, sd_beta, var_z, alpha)]


def _list2tensor(_list):
    return torch.Tensor(np.stack(_list, axis=1))


def get_init_params(Xs, F, Y, likelihood, n_trials):
    Xm = np.concatenate([x.mean(0, keepdims=True) for x in Xs], axis=0)
    Fe, Ye = F.numpy(), Y.long().numpy()

    if likelihood == "binomial":
        Xm = (Xm - Xm.mean(0, keepdims=True)) / xgower_factor(Xm)

        if n_trials == 2:
            Xm, Fe = Xm.repeat(2, axis=0), Fe.repeat(2, axis=0)
            to_expanded = np.array(([[0, 0], [1, 0], [1, 1]]))
            Ye = to_expanded[Y.long().numpy().T].transpose(1, 2, 0).reshape(-1, Y.shape[1])

        mu_z, sd_z, var_z, alpha = get_binomial_init_params(Xm, Fe, Ye)

    elif likelihood == "categorical":
        Xm, b, Fiv = regressOut(Xm, Fe, return_b=True, return_pinv=True)
        Xm = (Xm - Xm.mean(0, keepdims=True)) / (Xm.std(0, keepdims=True) * np.sqrt(Xm.shape[-1]))

        mu_z, sd_z, var_z, alpha = get_lr_init_params(Xm, Ye, b, Fiv)

    return mu_z, sd_z, var_z, alpha
