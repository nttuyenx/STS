from keras.engine.topology import Layer, InputSpec
from keras import backend as K
import numpy as np

class TemporalMeanPooling(Layer):
	"""
	This pooling layer accepts the temporal sequence output by a recurrent layer 
        and performs temporal pooling, looking at only the non-masked portion of the sequence.
        The pooling layer converts the entire variable-length hidden vector sequence into
        a single hidden vector.
	
	input shape: (nb_samples, nb_timesteps, nb_features)
	output shape: (nb_samples, nb_features)
        # https://github.com/fchollet/keras/issues/2151
        # http://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/36524166
	"""
	def __init__(self, **kwargs):
		super(TemporalMeanPooling, self).__init__(**kwargs)
		self.supports_masking = True
		self.input_spec = [InputSpec(ndim=3)]

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

	def call(self, x, mask=None): #mask: (nb_samples, nb_timesteps)
		if mask is None:
			mask = K.mean(K.ones_like(x), axis=-1)
		ssum = K.sum(x, axis=-2) #(nb_samples, np_features)
		mask = K.cast(mask, K.floatx())
		rcnt = K.sum(mask, axis=-1, keepdims=True) #(nb_samples)
		return ssum/rcnt

	def compute_mask(self, input, mask):
		return None

def cohyponyms():
    """
    Get cohyponyms from different sources such as PPDB, WordNet.
    Return a list of cohyponyms pairs.
    """
    cohyp_path = "../linguistic_constraints/wordnet_cohyponyms.txt"
    cohyponyms = []
    with open(cohyp_path) as f:
        for i, line in enumerate(f):
            cohyponyms.append(tuple(line.strip('\n').split()))
    
    return cohyponyms

def synonyms():
    """
    Get synonyms from different sources such as PPDB, WordNet.
    Return a list of synonym pairs.
    """
    syn_path = "../linguistic_constraints/ppdb_synonyms.txt"
    synonyms = []
    with open(syn_path) as f:
        for i, line in enumerate(f):
            synonyms.append(tuple(line.strip('\n').split()))
    
    return synonyms

def antonyms():
    """
    Get antonyms from different sources such as PPDB, WordNet.
    Return a list of antonyms pairs.
    """
    ant_path = "../linguistic_constraints/wordnet_antonyms.txt"
    antonyms = []
    with open(ant_path) as f:
        for i, line in enumerate(f):
            antonyms.append(tuple(line.strip('\n').split()))
    
    return antonyms

def get_hyponyms(synset):
    """
    Get all hyponyms of a synset in WordNet.
    """
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    
    return hyponyms | set(synset.hyponyms())

def get_mostfreq():
    """
    Get 10k Google most frequest words in English.
    """
    datapath= "../resource/google10k_en-usa.txt"
    mostfreq = []
    with open(datapath, 'r') as f:
        for i, line in enumerate(f):
            mostfreq.append(line.strip('\n'))
    print(len(mostfreq))
    
    return mostfreq
