"""
The hyperparameters for a model and arguments like model type, model name, data path, logs path, etc are defined here.
They can be changed by calling flags in the commandline arguements.

Required arguements:
1) model_type: which model you wish to train with. Valid model types: cbow, lstm.
2) model_name: the name assigned to the model being trained, this will prefix the name of the checkpoint files.
"""

import argparse
import io
import os

parser = argparse.ArgumentParser()

models = ['cbow', 'lstm']
def types(s):
    options = [mod for mod in models if s in models]
    if len(options) == 1:
        return options[0]
    return s

parser.add_argument("model_type", choices=models, type=types, help="Give model type.")
parser.add_argument("model_name", type=str, help="Give model name, this will name checkpoints files.")

parser.add_argument("--datapath", type=str, default="../data")
parser.add_argument("--ckptpath", type=str, default="../logs")

parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for model")
parser.add_argument("--dropout_rate", type=float, default=0.15, help="Dropout rate in the model")
parser.add_argument("--batch_size", type=int, default=30, help="Batch size for model")
parser.add_argument("--seq_length", type=int, default=25, help="Max sentence length")
parser.add_argument("--emb_train", action='store_true', help="Call if you want to train your word embeddings.")

args = parser.parse_args()

def load_parameters():
    FIXED_PARAMETERS = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "train_path": "{}/sts-train.csv".format(args.datapath),
        "dev_path": "{}/sts-dev.csv".format(args.datapath),
        "embedding_path": "{}/emb/glove.840B.300d.txt".format(args.datapath),
        "ckpt_path": "{}".format(args.ckptpath),
        "max_words": 50000,
        "embedding_dim": 300,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "emb_train": args.emb_train
    }

    return FIXED_PARAMETERS
