""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.

		self.sigmoid_result = 1 / (1+np.exp(-Input))
		return self.sigmoid_result

	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta

		delta_sigma = np.multiply(self.sigmoid_result, (1 - self.sigmoid_result))
		return np.multiply(delta_sigma, delta.T)

	    ############################################################################
