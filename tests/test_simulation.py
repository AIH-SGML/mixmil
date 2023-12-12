import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score

from mixmil import MixMIL
from mixmil.data import load_data


def calc_metrics(model, X, u, w):
    u_pred = model.predict(X["test"]).cpu().numpy().ravel()
    w_pred = model.get_weights(X["test"])[0].cpu().numpy().ravel()
    rho_bag = st.spearmanr(u_pred, u["test"]).correlation  # bag level correlation
    is_top_instance = (w["test"] > np.quantile(w["test"], 0.90)).long().ravel()
    auc_instance = roc_auc_score(is_top_instance, w_pred)  # instance-retrieval AUC
    return rho_bag, auc_instance


def test_simulation():
    X, F, Y, u, w = load_data(P=1, seed=0)
    model = MixMIL.init_with_mean_model(X["train"], F["train"], Y["train"], likelihood="binomial", n_trials=2)

    start_rho_bag, start_auc_instance = calc_metrics(model, X, u, w)
    model.train(X["train"], F["train"], Y["train"], n_epochs=40)
    end_rho_bag, end_auc_instance = calc_metrics(model, X, u, w)

    # assert that both metrics improved by at least 10%
    assert end_rho_bag > start_rho_bag * 1.1
    assert end_auc_instance > start_auc_instance * 1.1
