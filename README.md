# Baseline Models for STS Benchmark Dataset

This is the code We used to establish baselines for the STS Benchmark Dataset introduced in [SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation](http://www.aclweb.org/anthology/S/S17/S17-2001.pdf).

## Data
The dataset can be downloaded [here](http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz).

## Models
We present two baseline neural network models:

- Continuous Bag of Words (CBOW): in this model, each sentence is represented as the sum of the embedding representations of its words. This representation is passed to a fully-connected neural network. Code for this model is in [`cbow.py`](https://github.com/mikeng8/STS/blob/master/src/models/cbow.py)
- Bi-directional LSTM: in this model, the average of the states of a bidirectional LSTM or the final state is used as the sentence representation. Code for this model is in [`lstm.py`](https://github.com/mikeng8/STS/blob/master/src/models/lstm.py)

We use dropout and L2 for regularization the models.

## Training

### Command line flags

To start training there are two required command-line flags and other optional flags. All flags can be found in [`parameters.py`](https://github.com/mikeng8/STS/blob/master/src/util/parameters.py).

Required flags:

- `model_type`: there are two model types in this repository `cbow` and `lstm`. Must state which model to use.
- `model_name`: this is the experiment name. This name will be used the prefix the checkpoint files. 

Optional flags:

- `datapath`: path to the directory of the dataset, default is set to "../data".
- `ckptpath`: path to the directory where to store checkpoint files, default is set to "../logs".
- `learning_rate`: the learning rate to use during training, default value is set to 0.0001.
- `dropout_rate`: the hyperparameter for dropout-rate, default value is set to 0.15.
- `batch_size`: the hyperparameter for batch size, default value is set to 30.
- `seq_length`: the maximum sentence length, default value is set to 25. Sentences shorter than `seq_length` are padded and longer than `seq-length` are truncated.
- `emb_train`: boolean flag that determines if the model updates word embeddings during training. If called, the word embeddings are trainable. 

### Other parameters

Other parameters like the size of hidden layers, vocabulary size, etc can be changed directly in `parameters.py`.

### Sample commands
To execute all of the following sample commands, you must be in the "src" folder,

- To train on CBOW model, a sample command: <br/>
`python train.py cbow model-00 --seq_length 22 --batch_size 128 --dropout_rate 0.1 --emb_train`
- To train on LSTM model, a sample command: <br/>
`python train.py lstm model-01 --learning_rate 0.05 --seq_length 30`

while the `model_type` flag is fixed to `cbow` or `lstm`, the `model_name` flag can be changed.
