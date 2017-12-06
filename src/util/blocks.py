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
