# MixMIL
Code for the paper: [Mixed Models with Multiple Instance Learning](https://arxiv.org/abs/2311.02455)

Accepted at AISTATS 24 as an oral presentation & [Outstanding Student Paper Highlight](https://aistats.org/aistats2024/awards.html).

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

Make sure the `experiments` requirements are installed:
```
pip install "mixmil[experiments]"
```
### Histopathology
The histopathology experiment was performed on the [CAMELYON16](https://camelyon16.grand-challenge.org/) dataset.
#### Download Data
To download the embeddings provided by the DSMIL authors, either:
- Full embeddings: `python scripts/dsmil_data_download.py`
- PCA reduced embeddings: [Google Drive](https://drive.google.com/drive/folders/1X9ho1_W5ixyHSw_2hCfQsBb5nzkjMviA?usp=sharing)

### Microscopy 
The full BBBC021 dataset can be downloaded [here](https://bbbc.broadinstitute.org/BBBC021). 
#### Download Data
- We make the featurized cells available at [BBBC021](https://drive.google.com/file/d/1LEW74HUaJ2BMlPMmUlYMrsCFTmbpR2Rd/view?usp=drive_link)
- The features are stored as an [AnnData](https://anndata.readthedocs.io/en/latest/) object. We recommend using the [scanpy](https://scanpy.readthedocs.io/en/stable/) package to read and process them
- The weights of the featurizer trained with the SimCLR algorithm can be downloaded from the original [GitHub repository](https://github.com/SamriddhiJain/SimCLR-for-cell-profiling?tab=readme-ov-file)

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
