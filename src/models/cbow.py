import os
import util.parameters as params
from util.preprocessing import *
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras import layers, initializers, Input
from keras.layers import Dense, Embedding, Lambda, Dropout 
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers, regularizers
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

        print("Loading embeddings...")
        vocab_size = min(max_words, len(word_index)) + 1
        embedding_matrix = build_emb_matrix(FIXED_PARAMETERS["embedding_path"], vocab_size, word_index)
 
        embedding_layer = Embedding(vocab_size, 300,
                                    weights=[embedding_matrix],
                                    input_length=max_len,
                                    trainable=emb_train,
                                    name='VectorLookup')

        sent1_seq_in = Input(shape=(max_len,), dtype='int32', name='sent1_seq_in')
        embedded_sent1 = embedding_layer(sent1_seq_in)
        embedded_sent1_drop = layers.Dropout(dropout_rate)(embedded_sent1)
        encoded_sent1 = Lambda(lambda x: K.sum(x, axis=1))(embedded_sent1_drop)

        sent2_seq_in = Input(shape=(max_len,), dtype='int32', name='sent2_seq_in')
        embedded_sent2 = embedding_layer(sent2_seq_in)
        embedded_sent2_drop = layers.Dropout(dropout_rate)(embedded_sent2)
        encoded_sent2 = Lambda(lambda x: K.sum(x, axis=1))(embedded_sent2_drop)

        mul = layers.Multiply()([encoded_sent1, encoded_sent2])
        sub = layers.Subtract()([encoded_sent1, encoded_sent2])
        dif = Lambda(lambda x: K.abs(x))(sub)

        concatenated = layers.concatenate([mul, dif], axis=-1)

        x = Dense(150, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                bias_initializer=initializers.RandomNormal(stddev=0.1))(concatenated)
        x = Dropout(dropout_rate)(x)
        x = Dense(6, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                bias_initializer=initializers.RandomNormal(stddev=0.1))(x)

        gate_mapping = K.variable(value=np.array([[0.], [1.], [2.], [3.], [4.], [5.]]))
        preds = Lambda(lambda a: K.dot(a, gate_mapping), name='Prediction')(x)

        model = Model([sent1_seq_in, sent2_seq_in], preds)
        model.summary()

        def pearson(y_true, y_pred):
            """
            Pearson product-moment correlation metric.
            """
            return pearsonr(y_true, y_pred)

        early_stopping = EarlyStopping(monitor='pearson', patience=20, mode='max')
        # checkpointer = ModelCheckpoint(filepath=os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + '.hdf5',
        #                               verbose=1,
        #                               monitor='val_pearson',
        #                               save_best_only=True,
        #                               mode='max')

        Adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

        model.compile(optimizer=Adam,
                    loss='mse',
                    metrics=[pearson])

        history = model.fit([sent1_train_seq, sent2_train_seq], labels_train,
                            verbose=1,
                            epochs=300,
                            batch_size=batch_size,
                            callbacks=[early_stopping],
                            validation_data=([sent1_dev_seq, sent2_dev_seq], labels_dev))

        #filepath = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + '.hdf5'
        #model.save(filepath)
