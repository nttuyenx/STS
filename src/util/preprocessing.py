# -*- coding: utf-8 -*-,
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

def pearsonr(x, y):
    """
    Pearson correlation between predictions and labels.
    """
    xm = x - K.mean(x)
    ym = y - K.mean(y)
    r_num = K.sum(xm * ym)
    xs = K.sum(K.pow(xm, 2))
    ys = K.sum(K.pow(ym, 2))
    r_den = K.sqrt(xs) * K.sqrt(ys)
    r = r_num / r_den

    return r

def load_sts_data(path):
    """
    Load STS benchmark data.
    """
    genres, sent1, sent2, labels, scores = [], [], [], [], []
    for line in open(path):
        genre = line.split('\t')[0].strip()
        filename = line.split('\t')[1].strip()
        year = line.split('\t')[2].strip()
        other = line.split('\t')[3].strip()
        score = line.split('\t')[4].strip()
        s1 = line.split('\t')[5].strip()
        s2 = line.split('\t')[6].strip()
        label = float(score)
        genres.append(genre)
        sent1.append(s1)
        sent2.append(s2)
        labels.append(label)
        scores.append(score)
    labels = (np.asarray(labels)).flatten()

    return genres, sent1, sent2, labels, scores

def encode_labels(labels):
    """
    Encode labels Tai et al., 2015
    """
    labels_to_probs = []
    for label in labels:
        tmp = np.zeros(6, dtype=np.float32)
        if (int(label)+1 > 5):
            tmp[5] = 1
        else:
            tmp[int(label)+1] = label - int(label)
            tmp[int(label)] = int(label) - label + 1
        labels_to_probs.append(tmp)
    
    return np.asarray(labels_to_probs)
 
def tokenizing_and_padding(data_path, tokenizer, max_len):
    """
    Tokenizing and padding sentences to max length.
    """
    genres, sent1, sent2, labels, scores = load_sts_data(data_path)
    sent1_seq = tokenizer.texts_to_sequences(sent1)
    sent2_seq = tokenizer.texts_to_sequences(sent2)
    sent1_seq_pad = pad_sequences(sent1_seq, max_len)
    sent2_seq_pad = pad_sequences(sent2_seq, max_len)
    print('Shape of data tensor:', sent1_seq_pad.shape)
    print('Shape of label tensor:', labels.shape)
    
    return sent1_seq_pad, sent2_seq_pad, labels

def build_emb_matrix(emb_path, vocab_size, word_index):
    """
    Load word embeddings.
    """
    embedding_matrix = np.zeros((vocab_size, 300), dtype='float32')
    with open(emb_path, 'r') as f:
        for i, line in enumerate(f):
            s = line.split(' ')
            if (s[0] in word_index) and (word_index[s[0]] < vocab_size):
                embedding_matrix[word_index[s[0]], :] = np.asarray(s[1:])

    return embedding_matrix
