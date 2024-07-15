import numpy as np
import pytest
import torch

from mixmil import MixMIL


@pytest.fixture
def mock_data_binomial():
    Xs = [torch.randn(10, 3) for _ in range(5)]  # List of tensors
    F = torch.randn(5, 4)  # Fixed effects
    Y = torch.randint(0, 2, (5, 1))  # Labels for binomial
    return Xs, F, Y


@pytest.fixture
def mock_data_categorical():
    N, Q, K = 50, 10, 4
    bag_sizes = torch.randint(5, 15, (N,))
    Xs = [torch.randn(bag_sizes[n], Q) for n in range(N)]  # List of tensors
    F = torch.randn(N, K)  # Fixed effects
    Y = torch.randint(0, 5, (N, 1))  # Labels for categorical
    return Xs, F, Y


def test_init_with_mean_model_binomial(mock_data_binomial):
    Xs, F, Y = mock_data_binomial
    model = MixMIL.init_with_mean_model(Xs, F, Y, likelihood="binomial", n_trials=2)
    model.train(Xs, F, Y, n_epochs=3)
    assert isinstance(model, MixMIL)
    assert model.likelihood_name == "binomial"
    assert model.n_trials == 2
    assert model.log_sigma_u.numel() == 1


def test_init_with_mean_model_categorical(mock_data_categorical):
    Xs, F, Y = mock_data_categorical
    model = MixMIL.init_with_mean_model(Xs, F, Y, likelihood="categorical")
    model.train(Xs, F, Y, n_epochs=3)
    assert isinstance(model, MixMIL)
    assert model.likelihood_name == "categorical"
    assert model.n_trials is None
    assert model.log_sigma_u.numel() == len(np.unique(Y))  # separate prior for each class


def test_initialization():
    model = MixMIL(Q=10, K=5, P=2, likelihood="binomial", n_trials=2)
    assert model.Q == 10
    assert model.alpha.shape == (5, 2)


@pytest.mark.parametrize(
    "Q, K, P, likelihood, n_trials, mean_field",
    [
        (10, 5, 2, "categorical", None, True),
        (10, 5, 2, "categorical", None, False),
        (10, 5, 2, "binomial", 1, True),
        (10, 5, 2, "binomial", 2, False),
        (10, 5, 1, "binomial", 2, False),
    ],
)
def test_init_model(Q, K, P, likelihood, n_trials, mean_field):
    MixMIL(Q=Q, K=K, P=P, likelihood=likelihood, n_trials=n_trials, mean_field=mean_field)
