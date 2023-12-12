# MixMIL
Code for the paper: [Attention-based Multi-instance Mixed Models](https://arxiv.org/abs/2311.02455)

Please raise an issue for questions and bug-reports.
## Installation
Install with:
```
pip install mixmil
```
alternatively, if you want to include the optional experiment and test dependencies use:
```
pip install "mixmil[experiments,test]"
```
or if you want to adapt the code:
```
git clone https://github.com/AIH-SGML/mixmil.git
cd mixmil
pip install -e ".[experiments,test]"
```
To enable computations on GPU please follow the installation instructions of [PyTorch](https://pytorch.org/) and [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter).
MixMIL works e.g. with PyTorch 2.1.
## Experiments
See the notebooks in the `experiments` folder for examples on how to run the simulation and histopathology experiments.

### Histopathology
Install `anndata` (`pip install anndata`) to run the notebook. 

#### Download Data
To download the embeddings provided by the DSMIL authors, either:
- Full embeddings: `python scripts/dsmil_data_download.py`
- PCA reduced embeddings: [Google Drive](https://drive.google.com/drive/folders/1X9ho1_W5ixyHSw_2hCfQsBb5nzkjMviA?usp=sharing)


## Citation
```
@misc{engelmann2023attentionbased,
      title={Attention-based Multi-instance Mixed Models}, 
      author={Jan P. Engelmann and Alessandro Palma and Jakub M. Tomczak and Fabian J Theis and Francesco Paolo Casale},
      year={2023},
      eprint={2311.02455},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```