import numpy as np
import scipy.stats as st
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import trange

from mixmil import MixMIL
from mixmil.data import MILDataset, load_data


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


def train(X, F, Y, u, w, n_epochs=2_000, device="cpu"):
    X, F, Y = [
        {key: [x.to(device) for x in val] if isinstance(val, list) else val.to(device) for key, val in el.items()}
        for el in [X, F, Y]
    ]
    train_loader = DataLoader(MILDataset(X["train"], F["train"], Y["train"]), shuffle=True, batch_size=64)
    model = MixMIL.init_with_mean_model(X["train"], F["train"], Y["train"]).to(device)
    optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

    print("[START]\n", calc_metrics(model, X, u, w))

    for _ in trange(1, n_epochs + 1):
        for xs, f, y in train_loader:
            kld_w = len(xs) / len(Y["train"])
            pred = model(xs)
            loss = model.loss(pred, f, y, kld_w=kld_w)
            optim.zero_grad()
            loss.backward()
            optim.step()

    print("[END]\n", calc_metrics(model, X, u, w))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # embeddings, fixed effects, labels, bag predictions, instance weights
    X, F, Y, u, w = load_data()

    train(X, F, Y, u, w, n_epochs=150, device=device)

    for p in range(Y["train"].shape[1]):
        _Y, _u, _w = ({k: v[..., [p]] for k, v in el.items()} for el in [Y, u, w])
        train(X, F, _Y, _u, _w, n_epochs=150, device=device)
