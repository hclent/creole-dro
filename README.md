## [UNDER CONSTRUCTION] "On Language Models for Creoles"

This is the associated code for the CoNLL 2021 paper "On Language Models for Creoles"


### Environment Setup

```
conda create -n creole python=3.6
```

Environment requirements:

```
torch==1.8.0
transformers==4.4.2
pytorch-lightning
docopt
logzero
spacy
wilds==1.0.0
```

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

### Data

Naija:
[Pidgin UNMT Corpus](https://github.com/keleog/PidginUNMT/tree/master/corpus)
[Naija SUD](https://github.com/surfacesyntacticud/SUD_Naija-NSC)

Singlish:
[Singlish SMS Corpus](https://datasetsearch.research.google.com/search?query=singapore&docid=kqHXm0QYCrFZ229DAAAAAA%3D%3D)
[Singlish UD data](https://github.com/wanghm92/Sing_Par)

Haitian: 
[CMU Parallel Haitian-English Data](http://www.speech.cs.cmu.edu/haitian/text/)

If you would like access to the Haitian Disaster Response Corpus, please contact the original authors of this work. 

### Data pre-processing

Edit `main()` of `make_datasets.py`, as needed. 

**TODO**: edit this to have options, and relative paths. 

### Baseline pipeline

See `/experiments` for examples on how to train experiments. See `/experiments/creole-only/eval` for examples of evaluating the systems. 


