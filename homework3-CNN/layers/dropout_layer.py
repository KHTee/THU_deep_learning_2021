""" Dropout Layer """

import numpy as np

class DropoutLayer():
	def __init__(self):
		self.trainable = False

	def forward(self, Input, is_training=True):

		############################################################################
	    # TODO: Put your code here
		
		self.mask = None
		output = None
		self.input = Input
		self.drop_prob = 0.3
		self.is_training = is_training
		
		if is_training:
			self.mask = (np.random.uniform(*self.input.shape) > self.drop_prob)/ self.drop_prob
			output = np.multiply(self.mask, self.input)
		else:
			output = self.input

		output = output.astype(self.input.dtype, copy=False)

		return output
	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here

		delta_prev = None

		if self.is_training:
			delta_prev = np.multiply(delta, self.mask)
		else:
			delta_prev = delta

		return delta_prev
	    ############################################################################
