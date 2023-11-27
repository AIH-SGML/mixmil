import numpy as np
import scipy.stats as st
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import trange

from mixmil import MixMIL
from mixmil.data import MILDataset, load_data, mil_collate_fn


def calc_metrics(model, X, more):
    bag_preds = model.predict(X["test"])
    rho_bag = st.spearmanr(bag_preds, more["xmil"]["test"]).correlation

    pred_instance_weights, _ = model.get_weights(X["test"], ravel=True)
    is_top_instance = (more["w"]["test"].ravel() > np.quantile(more["w"]["test"].ravel(), 0.90)).long()
    instance_retreival_auc = roc_auc_score(is_top_instance, pred_instance_weights)
    return {"rho_bag": rho_bag, "auc_instance": instance_retreival_auc}


def train(X, Y, F, more, n_epochs=2_000, device="cpu"):
    X, F, Y = [
        {key: [x.to(device) for x in val] if isinstance(val, list) else val.to(device) for key, val in el.items()}
        for el in [X, F, Y]
    ]
    # collate_fn=mil_collate_fn,
    train_loader = DataLoader(MILDataset(X["train"], F["train"], Y["train"]), shuffle=True, batch_size=64)
    model = MixMIL.init_with_mean_model(X["train"], F["train"], Y["train"]).to(device)
    optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

    rdict = calc_metrics(model, X, more)
    print("[START]\n", rdict)
    H = []
    for epoch in trange(1, n_epochs + 1):
        rdict = calc_metrics(model, X, more)
        pred = model(X["train"])
        ldict = model.loss(pred, F["train"], Y["train"], return_all=True)
        rdict.update(ldict)
        H.append(rdict)
        for xs, f, y in train_loader:
            kld_w = len(xs) / len(Y["train"])
            pred = model(xs)
            loss = model.loss(pred, f, y, kld_w=kld_w)
            optim.zero_grad()
            loss.backward()
            optim.step()

    rdict = calc_metrics(model, X, more)
    print("[END]\n", rdict)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y, F, more = load_data()

    train(X, Y, F, more, n_epochs=200, device=device)
