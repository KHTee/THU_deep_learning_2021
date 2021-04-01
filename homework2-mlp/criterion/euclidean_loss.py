""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.

		num_train = logit.shape[0]
		self.delta = (logit - gt) / num_train 
		self.loss = np.linalg.norm(self.delta) 

		pred = np.zeros_like(gt)
		pred[np.arange(num_train), np.argmax(logit, axis=1)] = 1
		self.acc = float(np.sum(np.logical_and(pred, gt))/num_train)

	    ############################################################################

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)

		return (self.delta / self.loss)

	    ############################################################################
