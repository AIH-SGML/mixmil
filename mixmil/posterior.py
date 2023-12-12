import numpy as np
import torch
from torch.distributions import LowRankMultivariateNormal


def get_params(vc_mean, vc_sd, n_outs, n_vars, mean_field):
    mu_z = torch.Tensor(vc_mean)
    mu_u = torch.zeros_like(mu_z)
    mu = torch.cat([mu_u, mu_z], 1)
    sd_z = torch.Tensor(vc_sd)
    sd_u = torch.sqrt(0.1 * torch.ones_like(sd_z))

    if mean_field:
        cov_factor = torch.zeros(n_outs, n_vars, 1)
        cov_logdiag = 2.0 * torch.log(torch.cat([sd_u, sd_z], 1))
    else:
        diag = torch.diag_embed(torch.cat([sd_u, sd_z], 1))
        cov_factor = 1e-4 * torch.randn(diag.shape) + diag
        cov_logdiag = np.log(1e-4) * torch.ones(n_outs, n_vars)

    return mu, cov_factor, cov_logdiag


class GaussianVariationalPosterior(torch.nn.Module):
    def __init__(self, n_vars, n_outs, mean_field=True, init_params=None):
        super().__init__()
        self.n_vars = n_vars
        self.n_outs = n_outs
        self.mean_field = mean_field

        if init_params is not None:
            mu_z, sd_z, *_ = init_params
            mu_z = mu_z.T
            sd_z = sd_z.T
        else:
            mu_z = 1e-3 * torch.randn(n_outs, n_vars // 2)
            sd_z = 1e-3 * torch.randn(n_outs, n_vars // 2)

        mu, cov_factor, cov_logdiag = get_params(mu_z, sd_z, n_outs, n_vars, mean_field)

        self.mu = torch.nn.Parameter(mu)
        if mean_field:
            self.register_buffer("cov_factor", cov_factor)
            self.cov_logdiag = torch.nn.Parameter(cov_logdiag)
        else:
            self.cov_factor = torch.nn.Parameter(cov_factor)
            self.register_buffer("cov_logdiag", cov_logdiag)

    @property
    def distribution(self):
        return LowRankMultivariateNormal(self.mu, self.cov_factor, torch.exp(self.cov_logdiag))

    @property
    def q_mu(self):
        return self.mu.T

    def sample(self, n_samples):
        return self.distribution.rsample([n_samples]).permute([2, 1, 0])

    def extra_repr(self) -> str:
        return f"n_vars=2*{self.n_vars//2}, n_outs={self.n_outs}, mean_field={self.mean_field}"
