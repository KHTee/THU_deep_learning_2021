""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

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

		self.num_train = logit.shape[0]
		self.gt = gt
        
		self.probs = np.exp(logit)
		self.e_y = np.sum(np.multiply(self.probs,self.gt), axis=1)
		self.e_sum = np.sum(self.probs, axis=1)
		
		self.loss = np.sum(-np.log(self.e_y / self.e_sum))
		self.loss /= self.num_train

		# Accuracy
		pred = np.zeros_like(gt)
		pred[np.arange(self.num_train), np.argmax(logit, axis=1)] = 1
		self.acc = float(np.sum(np.logical_and(pred, gt)) / self.num_train)

	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)

		delta_score = self.probs / self.e_sum.reshape(self.num_train,1)
		delta_score -= self.gt

		return delta_score / self.num_train

	    ############################################################################
