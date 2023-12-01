import numpy as np
import scipy.stats as st
import torch
from sklearn.metrics import roc_auc_score

from mixmil import MixMIL
from mixmil.data import load_data


def calc_metrics(model, X, u, w):
    res_dict = {}
    for p in range(u["test"].shape[1]):
        u_pred = model.predict(X["test"])
        rho_bag = st.spearmanr(u_pred[..., p], u["test"][..., p]).correlation

        w_pred, _ = model.get_weights(X["test"], ravel=False)
        is_top_instance = (w["test"][..., p] > np.quantile(w["test"][..., p], 0.90)).long()
        instance_retreival_auc = roc_auc_score(is_top_instance.ravel(), w_pred[..., p].ravel())
        res_dict.update({f"rho_bag_{p}": rho_bag, f"auc_instance_{p}": instance_retreival_auc})
    return res_dict


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # embeddings, fixed effects, labels, bag predictions, instance weights
    X, F, Y, u, w = load_data()
    model = MixMIL.init_with_mean_model(X["train"], F["train"], Y["train"]).to(device)
    print(model)
    print("[START]\n", calc_metrics(model, X, u, w))
    X, F, Y = [{key: val.to(device) for key, val in el.items()} for el in [X, F, Y]]
    model.train(X["train"], F["train"], Y["train"], n_epochs=150)
    print("[END]\n", calc_metrics(model, X, u, w))
