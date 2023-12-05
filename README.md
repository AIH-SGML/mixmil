# MixMIL

## Installation
```
pip install -e .
```
To enable computations on GPU please follow the installation instructions of [PyTorch](https://pytorch.org/) and [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter).
MixMIL works with PyTorch 2.1.
## Experiments
See the notebooks in the `experiments` folder for examples on how to run the simulation and histopathology experiments.

## Download Histopathology Data
To download the embeddings provided by the DSMIL authors, please run:
```
python scripts/dsmil_data_download.py
```

