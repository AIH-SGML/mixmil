import numpy as np
import torch
from torch.distributions import Binomial, Categorical, LowRankMultivariateNormal
from torch.distributions.kl import kl_divergence
from torch_scatter import scatter_softmax, segment_add_csr

from mixmil.data import setup_scatter
from mixmil.posterior import GaussianVariationalPosterior
from mixmil.utils import get_init_params


class MixMIL(torch.nn.Module):
    """Attention-based Multi-instance Mixed Models

    https://arxiv.org/abs/2311.02455

    """

    def __init__(self, Q, K, P=1, likelihood="binomial", n_trials=2, mean_field=False, init_params=None):
        """
        Initialize the MixMil class.

        Parameters:
        - Q (int): The dimension of the latent space.
        - K (int): The number of fixed effects.
        - P (int): The number of outputs.
        - mean_field (bool, optional): Toggle mean field approximation for the posterior. Default is False.
        """
        super().__init__()
        self.Q = Q

        alpha = torch.zeros((K, P))
        log_sigma_u = torch.full((1, P), 0.5 * np.log(0.5))
        log_sigma_z = torch.full((1, P), 0.5 * np.log(0.5))

        if init_params is not None:
            *_, var_z, alpha = init_params
            log_sigma_z = 0.5 * torch.log(var_z)

        self.alpha = torch.nn.Parameter(alpha)
        self.log_sigma_u = torch.nn.Parameter(log_sigma_u)
        self.log_sigma_z = torch.nn.Parameter(log_sigma_z)

        self.posterior = GaussianVariationalPosterior(2 * Q, P, mean_field, init_params)

        self.likelihood_name = likelihood
        self.n_trials = n_trials if likelihood == "binomial" else None

    def init_with_mean_model(Xs, F, Y, likelihood="binomial", n_trials=2, mean_field=False):
        assert (likelihood == "binomial" and n_trials is not None and 0 < n_trials <= 2) or (
            likelihood == "categorical"
        ), f"n_trials must be 1 or 2 to initialize with binomial mean model, got {n_trials=} and {likelihood=}"
        init_params = get_init_params(Xs, F, Y, likelihood, n_trials)
        Q, K, P = Xs[0].shape[1], F.shape[1], Y.shape[1]
        return MixMIL(Q, K, P, likelihood, n_trials, mean_field, init_params)

    @property
    def prior_distribution(self):
        device = self.log_sigma_u.device
        scale_u = self.log_sigma_u.T * torch.ones([1, self.Q], device=device)
        scale_z = self.log_sigma_z.T * torch.ones([1, self.Q], device=device)
        cov_logdiag = 2 * torch.cat([scale_u, scale_z], 1)
        cov_factor = torch.zeros_like(cov_logdiag)[:, :, None]
        mu = torch.zeros_like(cov_logdiag)
        return LowRankMultivariateNormal(mu, cov_factor, torch.exp(cov_logdiag))

    @property
    def posterior_distribution(self):
        return self.posterior.distribution

    @property
    def qu_mu(self):
        return self.posterior.q_mu[: self.Q]

    @property
    def qz_mu(self):
        return self.posterior.q_mu[self.Q :]

    def likelihood(self, logits, y):
        if self.likelihood_name == "binomial":
            return Binomial(total_count=self.n_trials, logits=logits).log_prob(y[:, :, None]).sum(1).mean()
        elif self.likelihood_name == "categorical":
            return Categorical(logits=logits.permute(0, 2, 1)).log_prob(y).mean()

    def loss(self, xmil, f, y, kld_w=1.0, return_all=False):
        logits = f.mm(self.alpha)[:, :, None] + xmil

        ll = self.likelihood(logits, y)
        kld = kl_divergence(self.posterior_distribution, self.prior_distribution)
        kld_term = kld_w * kld.sum() / y.shape[0]
        loss = -ll + kld_term
        if return_all:
            return dict(loss=loss, ll=ll, kld=kld_term)
        return loss

    def get_betas(self, n_samples=None, predict=False):
        assert not (n_samples and predict)
        if n_samples:
            beta = self.posterior.sample(n_samples)
            beta_u = beta[: self.Q, :, :]
            beta_z = beta[self.Q :, :, :]
        else:
            beta_u = self.qu_mu[:, :, None]
            beta_z = self.qz_mu[:, :, None]
        return beta_u, beta_z

    def calc_xmil(self, Xs, n_samples=8, scaling=None, predict=False):
        beta_u, beta_z = self.get_betas(n_samples, predict)
        b = torch.sqrt((beta_z**2).mean(0, keepdim=True))
        eta = beta_z / b

        if torch.is_tensor(Xs):
            xmil = self._calc_xmil_tensor(beta_u, eta, Xs)
        else:
            xmil = self._calc_xmil_scatter(beta_u, eta, Xs)

        mean, std = (xmil.mean(0), xmil.std(0)) if scaling is None else scaling
        if std.isnan().any():
            std = 1
        xmil = b * (xmil - mean) / std
        return xmil

    def forward(self, *args, **kwargs):
        return self.calc_xmil(*args, **kwargs)

    def _calc_xmil_tensor(self, beta_u, eta, Xs):
        u = torch.einsum("niq,qps->nips", Xs, beta_u)
        w = torch.softmax(u, dim=1)
        z = torch.einsum("niq,qps->nips", Xs, eta)
        xmil = torch.einsum("nips,nips->nps", w, z)
        return xmil

    def _calc_xmil_scatter(self, beta_u, eta, Xs):
        x, i, i_ptr = setup_scatter(Xs)

        u = torch.einsum("iq,qps->ips", x, beta_u)
        w = scatter_softmax(u, i, dim=0)
        z = torch.einsum("iq,qps->ips", x, eta)
        xmil = segment_add_csr(w * z, i_ptr)
        return xmil

    @torch.inference_mode()
    def predict(self, Xs):
        return self.calc_xmil(Xs, n_samples=None, predict=True).squeeze(2)

    @torch.inference_mode()
    def get_weights(self, Xs, ravel=False):
        beta_u, _ = self.get_betas(predict=True)
        if torch.is_tensor(Xs):
            u = torch.einsum("niq,qps->nips", Xs, beta_u)
            w = torch.softmax(u, dim=1)

        else:
            x, i, _ = self.setup_scatter(Xs)
            u = torch.einsum("iq,qps->ips", x, beta_u)
            w = scatter_softmax(u, i, dim=0)

        if ravel:
            w, u = w.ravel(), u.ravel()
        return w, u


if __name__ == "__main__":
    I = 10
    N = 50
    Q = 30
    P = 3
    K = 2
    Xsl = [torch.rand(I, Q) for _ in range(N)]
    Xst = torch.cat([t.reshape(1, I, Q) for t in Xsl], dim=0)
    F = torch.rand(N, K)
    Y = torch.randint(0, 2, (N, P))
    model = MixMIL.init_with_mean_model(Xst, F, Y, likelihood="binomial", mean_field=False)
    model.predict(Xst)
    pred = model(Xst)
    loss = model.loss(pred, F, Y)
    loss.backward()

    model = MixMIL.init_with_mean_model(Xsl, F, Y, likelihood="binomial", mean_field=False)
    model.predict(Xsl)
    pred = model(Xsl)
    loss = model.loss(pred, F, Y)
    loss.backward()

    I = 10
    N = 50
    Q = 30
    P = 1
    K = 1
    Xsl = [torch.rand(I, Q) for _ in range(N)]
    Xst = torch.cat([t.reshape(1, I, Q) for t in Xsl], dim=0)
    F = torch.rand(N, K)
    Y = torch.randint(0, 5, (N, 1))
    model = MixMIL.init_with_mean_model(Xst, F, Y, likelihood="categorical", mean_field=False)
    model.predict(Xst)
    pred = model(Xst)
    loss = model.loss(pred, F, Y)
    loss.backward()
    print("DONE")
