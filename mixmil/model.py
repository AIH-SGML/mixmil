import numpy as np
import torch
from torch.distributions import Binomial, Categorical, LowRankMultivariateNormal, Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
from torch_scatter import scatter_softmax, segment_add_csr
from tqdm.auto import trange

from mixmil.data import MILDataset, mil_collate_fn, setup_scatter
from mixmil.posterior import GaussianVariationalPosterior
from mixmil.utils import get_init_params


class MixMIL(torch.nn.Module):
    """Attention-based Multi-instance Mixed Models

    https://arxiv.org/abs/2311.02455

    """

    def __init__(
        self,
        Q,
        K,
        P=1,
        likelihood="binomial",
        n_trials=2,
        mean_field=False,
        init_params=None,
        pos_weight=None,
        standardize="batch",
        eps=1e-6,
    ):
        r"""Initialize the MixMil class.

        Parameters:
        - Q (int): The dimension of the latent space.
        - K (int): The number of fixed effects.
        - P (int): The number of outputs.
        - likelihood (str, optional): The likelihood to use. Either "binomial" or "categorical". Default is "binomial".
        - n_trials (int, optional): Number of trials for binomial likelihood. Not used for categorical. Default is 2.
        - mean_field (bool, optional): Toggle mean field approximation for the posterior. Default is False.
        - init_params (tuple, optional): Tuple of (mean, var, var_z, alpha) to initialize the model. Default is None.
            mean (torch.Tensor): The mean of the posterior. Shape: (Q, P). d
            var (torch.Tensor): The variance of the posterior. Shape: (Q, P).
            var_z (torch.Tensor): The $\sigma_{\beta}^2$ hparam of the prior.
                Shape: (1, P) with separate and (1, 1) with shared priors .
            alpha (torch.Tensor): The fixed effect parameters. Shape: (K, P).
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
        self.is_trained = False
        # Introduce a scale parameter for Normal likelihood
        if self.likelihood_name == "normal":
            self.log_scale = torch.nn.Parameter(torch.tensor(0.0))

        self.pos_weight = pos_weight

        assert standardize in ("batch", "fixed", "none")
        self.standardize = standardize
        self._scaling = None
        self.eps = eps

    def init_with_mean_model(Xs, F, Y, likelihood="binomial", n_trials=None, mean_field=False, **kwargs):
        assert (likelihood == "binomial" and n_trials is not None and 0 < n_trials <= 2) or (
            likelihood in ["categorical", "normal"] and n_trials is None
        ), f"n_trials must be 1 or 2 for binomial. For {likelihood}, n_trials must be None."
        init_params = get_init_params(Xs, F, Y, likelihood, n_trials)
        Q, K, P = Xs[0].shape[1], F.shape[1], init_params[0].shape[1]
        return MixMIL(Q, K, P, likelihood, n_trials, mean_field, init_params, **kwargs)

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
            if self.pos_weight is not None:
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
                return -criterion(logits, y[:, :, None].expand(-1, -1, logits.shape[-1])).sum(1).mean()
            return Binomial(total_count=self.n_trials, logits=logits).log_prob(y[:, :, None]).sum(1).mean()
        elif self.likelihood_name == "categorical":
            logits = logits.permute(0, 2, 1)
            if logits.shape[-1] == 1:
                logits = torch.cat([-logits, logits], 2)
            return Categorical(logits=logits).log_prob(y).mean()
        elif self.likelihood_name == "normal":
            scale = torch.exp(self.log_scale)
            return Normal(loc=logits, scale=scale).log_prob(y[:, :, None]).sum(1).mean()

    def loss(self, u, f, y, kld_w=1.0, return_dict=False):
        logits = f.mm(self.alpha)[:, :, None] + u

        ll = self.likelihood(logits, y)
        kld = kl_divergence(self.posterior_distribution, self.prior_distribution)
        kld_term = kld_w * kld.sum() / y.shape[0]
        loss = -ll + kld_term
        if return_dict:
            return loss, dict(loss=loss.item(), ll=ll.item(), kld=kld_term.item())
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

    @torch.inference_mode()
    def fit_scaling(self, Xs):
        """Compute global (fixed) scaling from the current posterior means."""
        beta_u, beta_z = self.get_betas(n_samples=None, predict=True)
        b = torch.sqrt((beta_z**2).mean(0, keepdim=True))
        eta = beta_z / b
        if torch.is_tensor(Xs):
            u_raw = self._calc_bag_emb_effect_tensor(beta_u, eta, Xs)
        else:
            u_raw = self._calc_bag_emb_effect_scatter(beta_u, eta, Xs)
        mean = u_raw.mean(0)
        std = u_raw.std(0).clamp_min(self.eps)
        self._scaling = (mean, std)

    def forward(self, Xs, n_samples=8, scaling=None, predict=False):
        beta_u, beta_z = self.get_betas(n_samples, predict)
        b = torch.sqrt((beta_z**2).mean(0, keepdim=True))
        eta = beta_z / b

        if torch.is_tensor(Xs):
            u_raw = self._calc_bag_emb_effect_tensor(beta_u, eta, Xs)
        else:
            u_raw = self._calc_bag_emb_effect_scatter(beta_u, eta, Xs)

        if scaling is not None:
            mean, std = scaling
            u = b * (u_raw - mean) / (std + self.eps)
        elif self.standardize == "batch":
            mean, std = u_raw.mean(0), u_raw.std(0)
            std = torch.where(torch.isnan(std), torch.ones_like(std), std).clamp_min(self.eps)
            u = b * (u_raw - mean) / std
        elif self.standardize == "fixed":
            if self._scaling is None:
                raise RuntimeError("standardize='fixed' but no scaling set. Call fit_scaling(Xs) first.")
            mean, std = self._scaling
            u = b * (u_raw - mean) / (std + self.eps)
        else:  # "none"
            u = b * u_raw

        return u

    def _calc_bag_emb_effect_tensor(self, beta_u, eta, Xs):
        _w = torch.einsum("niq,qps->nips", Xs, beta_u)
        w = torch.softmax(_w, dim=1)
        t = torch.einsum("niq,qps->nips", Xs, eta)
        u = torch.einsum("nips,nips->nps", w, t)
        return u

    def _calc_bag_emb_effect_scatter(self, beta_u, eta, Xs):
        x, i, i_ptr = setup_scatter(Xs)

        _w = torch.einsum("iq,qps->ips", x, beta_u)
        w = scatter_softmax(_w, i, dim=0)
        t = torch.einsum("iq,qps->ips", x, eta)
        u = segment_add_csr(w * t, i_ptr)
        return u

    def train(self, X, F, Y, n_epochs=2_000, batch_size=64, lr=1e-3, verbose=True, pos_weight=None):
        train_loader = DataLoader(
            MILDataset(X, F, Y),
            shuffle=True,
            batch_size=batch_size,
            collate_fn=None if torch.is_tensor(X) else mil_collate_fn,
        )
        optim = torch.optim.Adam(lr=lr, params=self.parameters())
        if pos_weight is not None:
            if torch.is_tensor(pos_weight):
                self.pos_weight = pos_weight
            else:
                num_pos = Y.sum().item()
                num_neg = Y.shape[0] - num_pos
                self.pos_weight = torch.tensor([num_neg / num_pos], dtype=F.dtype, device=F.device)

        history = []
        if self.standardize == "fixed":
            self.fit_scaling(X)
        for epoch in trange(1, n_epochs + 1, desc="Epoch", disable=not verbose):
            for step, (xs, f, y) in enumerate(train_loader):
                u = self(xs)
                loss, ldict = self.loss(u, f, y, kld_w=len(xs) / len(Y), return_dict=True)
                ldict["epoch"], ldict["step"] = epoch, step
                history.append(ldict)
                optim.zero_grad()
                loss.backward()
                optim.step()

        self.is_trained = True
        return history

    @torch.inference_mode()
    def predict(self, Xs, scaling=None):
        return self(Xs, n_samples=None, predict=True, scaling=scaling).squeeze(2)

    @torch.inference_mode()
    def get_weights(self, Xs, ravel=False):
        """Get instance weights after and before softmax"""
        beta_u, _ = self.get_betas(predict=True)
        beta_u = beta_u.squeeze(2)  # not taking mcmc samples
        if torch.is_tensor(Xs):
            _w = torch.einsum("niq,qp->nip", Xs, beta_u)
            w = torch.softmax(_w, dim=1)

        else:
            x, i, _ = setup_scatter(Xs)
            _w = torch.einsum("iq,qp->ip", x, beta_u)
            w = scatter_softmax(_w, i, dim=0)

        if ravel:
            w, _w = w.ravel(), _w.ravel()
        elif not torch.is_tensor(Xs):
            _w = [_w[i == idx] for idx in range(len(Xs))]
            w = [w[i == idx] for idx in range(len(Xs))]
        return w, _w

    def extra_repr(self):
        string = f"Q={self.Q}, K={self.alpha.shape[0]}, P={self.alpha.shape[1]}, likelihood={self.likelihood_name}"
        if self.likelihood_name == "binomial":
            string += f", n_trials={self.n_trials}"
        string += f", device={self.alpha.device}, trained={self.is_trained}"
        string += f"\n(alpha): Parameter(shape={tuple(self.alpha.shape)})\n"
        string += f"(log_sigma_u): Parameter(shape={tuple(self.log_sigma_u.shape)})\n"
        string += f"(log_sigma_z): Parameter(shape={tuple(self.log_sigma_z.shape)})"
        if self.likelihood_name == "normal":
            string += f"\n(log_scale): Parameter(shape={tuple(self.log_scale.shape)})"
        return string


if __name__ == "__main__":
    N = 100
    P = 1
    Q = 10
    K = 1
    I = 10
    Xs = [torch.randn(I, Q) for _ in range(N)]
    F = torch.randn(N, K)
    Y = torch.randint(0, 2, (N, 1)).float()
    model = MixMIL(Q, K, P, likelihood="binomial", n_trials=1, mean_field=False, init_params=None)
    model.train(Xs, F, Y, n_epochs=100, pos_weight=True)
