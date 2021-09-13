## [UNDER CONSTRUCTION] Understanding Creoles with DRO

This is the associated code for the CoNLL 2021 paper "On Language Models for Creoles"


### Setup

```
conda create -n creole python=3.6
```

Environment requirements:
torch==1.8.0
transformers==4.4.2
pytorch-lightning
docopt
logzero
spacy
wilds==1.0.0

Other setup:
```
python -m spacy download en_core_web_sm


pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

For example:


pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric
```

### Downloading the data


### Baseline pipeline


### Baseline backburner

Q: How can we get just Haitian out of MBERT to use as a Baseline?

