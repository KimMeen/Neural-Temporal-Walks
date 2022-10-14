# Neural-Temporal-Walks

## Introduction

[NIPS 2022] The official PyTorch implementation of "[Neural Temporal Walks: Motif-Aware Representation Learning on Continuous-Time Dynamic Graphs](https://mingjin.dev/assets/pdf/neurips-22-jin.pdf)"

## Authors
[Ming Jin](https://mingjin.dev/), [Yuan-Fang Li](https://users.monash.edu.au/~yli/), [Shirui Pan](https://shiruipan.github.io/)

## Requirements
- matplotlib==3.3.4
- numba==0.54.1
- numpy==1.19.2
- pandas==1.2.2
- scikit_learn==1.1.1
- torch==1.7.1
- torchdiffeq==0.2.2
- tqdm==4.59.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Dataset and preprocessing

### Option 1. Use our preprocessed datasets

- Click [here](https://drive.google.com/uc?export=download&id=1jao1WgVt6VKfDA4KNPfxqlH0CeBNNL6Y) to download our preprocessed datasets.
- Unzip the downloaded file
- Place all dataset files under the ./data directory

### Option 2. Preprocess datasets by yourself

- Please refer to our paper to download the raw datasets
- Put the dataset files under the ./data directory
- Use the provided scripts to preprocess the raw dataset ```.csv``` files. For example:
  - For the CollegeMsg dataset, you can run our script directly to preprocess it: ```python collegemsg.py```
  - For the Taobao dataset, you need to execute ```taobao_preprocessing.ipynb``` first before running ```python taobao.py```

## Model training

Here we provide three examples. Firstly, enter the directory with training scripts:

```cd scripts/```

To train on the CollegeMsg dastaset:
- Inductive: ```bash collegemsg_inductive.sh```
- Transductive: ```bash collegemsg_transductive.sh```


To train on the Enron dastaset:
- Inductive: ```bash enron_inductive.sh```
- Transductive: ```bash enron_transductive.sh```

To train on the Taobao dastaset:
- Inductive: ```bash taobao_inductive.sh```
- Transductive: ```bash taobao_transductive.sh```

## Detailed usage

Please refer to the function ```get_args()``` in ```util.py``` for the detailed description of each hyperparameter.


## Acknowledgement
Our implementation adapts the code of [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) and [CAWs](https://github.com/snap-stanford/CAW) as the code base and extensively adapts it to our purpose. We thank the authors for sharing their code.

## Cite us
If you find this research useful, please cite our paper:
```
@inproceedings{
jin2022neural,
title={Neural Temporal Walks: Motif-Aware Representation Learning on Continuous-Time Dynamic Graphs},
author={Ming Jin and Yuan-Fang Li and Shirui Pan},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022}
}
```