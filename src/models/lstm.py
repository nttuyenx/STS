import os
import util.parameters as params
from util.preprocessing import *
from util.blocks import TemporalMeanPooling
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras import Input, layers, optimizers, regularizers
from keras import backend as K
from keras.layers import Dense, Embedding, Lambda, Dropout 
from keras.layers import LSTM, Bidirectional
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

FIXED_PARAMETERS = params.load_parameters()

class STSModel(object):
    def __init__(self, max_len, emb_train):
        # Define hyperparameters
        modname = FIXED_PARAMETERS["model_name"]
        learning_rate = FIXED_PARAMETERS["learning_rate"]
        dropout_rate = FIXED_PARAMETERS["dropout_rate"]
        batch_size = FIXED_PARAMETERS["batch_size"]
        max_words = FIXED_PARAMETERS["max_words"]

        print("Loading data...")
        genres_train, sent1_train, sent2_train, labels_train_, scores_train = load_sts_data(FIXED_PARAMETERS["train_path"])
        genres_dev, sent1_dev, sent2_dev, labels_dev_, scores_dev = load_sts_data(FIXED_PARAMETERS["dev_path"])

        print("Building dictionary...")
        text = sent1_train + sent2_train + sent1_dev + sent2_dev
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(text)
        word_index = tokenizer.word_index
        
        print("Padding and indexing sentences...")
        sent1_train_seq, sent2_train_seq, labels_train = tokenizing_and_padding(FIXED_PARAMETERS["train_path"], tokenizer, max_len)
        sent1_dev_seq, sent2_dev_seq, labels_dev  = tokenizing_and_padding(FIXED_PARAMETERS["dev_path"], tokenizer, max_len)

        print("Encoding labels...")
        train_labels_to_probs = encode_labels(labels_train)
        dev_labels_to_probs = encode_labels(labels_dev)
        
        print("Loading embeddings...")
        vocab_size = min(max_words, len(word_index)) + 1
        embedding_matrix = build_emb_matrix(FIXED_PARAMETERS["embedding_path"], vocab_size, word_index)
    
        embedding_layer = Embedding(vocab_size, 300,
                                    weights=[embedding_matrix],
                                    input_length=max_len,
                                    trainable=emb_train,
                                    mask_zero=True,
                                    name='VectorLookup')

        lstm = Bidirectional(LSTM(150, #dropout=dropout_rate, recurrent_dropout=0.1,
                    return_sequences = False, kernel_regularizer=regularizers.l2(1e-4), name='RNN'))

        sent1_seq_in = Input(shape=(max_len,), dtype='int32', name='Sentence1')
        embedded_sent1 = embedding_layer(sent1_seq_in)
        encoded_sent1 = lstm(embedded_sent1)
        #mean_pooling_1 = TemporalMeanPooling()(encoded_sent1)
        #sent1_lstm = Dropout(0.1)(mean_pooling_1)

        sent2_seq_in = Input(shape=(max_len,), dtype='int32', name='Sentence2')
        embedded_sent2 = embedding_layer(sent2_seq_in)
        encoded_sent2 = lstm(embedded_sent2)
        #mean_pooling_2 = TemporalMeanPooling()(encoded_sent2)
        #sent2_lstm = Dropout(0.1)(mean_pooling_2)

        mul = layers.Multiply(name='S1.S2')([encoded_sent1, encoded_sent2]) #([mean_pooling_1, mean_pooling_2])
        sub = layers.Subtract(name='S1-S2')([encoded_sent1, encoded_sent2]) #([mean_pooling_1, mean_pooling_2])
        dif = Lambda(lambda x: K.abs(x), name='Abs')(sub)

        concatenated = concatenate([mul, dif], name='Concat')
        x = Dense(50, activation='sigmoid', name='Sigmoid', kernel_regularizer=regularizers.l2(1e-4))(concatenated)
        preds = Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(1e-4), name='Softmax')(x)

        model = Model([sent1_seq_in, sent2_seq_in], preds)
        model.summary()

        def pearson(y_true, y_pred):
            """
            Pearson product-moment correlation metric.
            """
            gate_mapping = K.variable(value=np.array([[0.], [1.], [2.], [3.], [4.], [5.]]))
            y_true = K.clip(y_true, K.epsilon(), 1)
            y_pred = K.clip(y_pred, K.epsilon(), 1)
            y_true = K.reshape(K.dot(y_true, gate_mapping), (-1,))
            y_pred = K.reshape(K.dot(y_pred, gate_mapping), (-1,))

            return pearsonr(y_true, y_pred)
        
        early_stopping = EarlyStopping(monitor='pearson', patience=10, mode='max')
        # checkpointer = ModelCheckpoint(filepath=os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + '.hdf5',
        #                                verbose=1,
        #                                monitor='val_pearson',
        #                                save_best_only=True,
        #                                mode='max')

        Adagrad = optimizers.Adagrad(lr=learning_rate)
        model.compile(optimizer=Adagrad, loss='kld', metrics=[pearson])

        history = model.fit([sent1_train_seq, sent2_train_seq], train_labels_to_probs,
                            epochs=30,
                            batch_size=batch_size,
                            shuffle=True,
                            callbacks=[early_stopping],
                            validation_data=([sent1_dev_seq, sent2_dev_seq], dev_labels_to_probs))

        #filepath = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + '.hdf5'
        #model.save(filepath)
