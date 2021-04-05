# -*- encoding: utf-8 -*-

import numpy as np

class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input, **kwargs):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
		Return:
			The tensor after being pooled.
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		self.input = Input
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)

		num_image, C, input_height, input_width = self.input.shape
		self.stride = 2
    
		H_new = int(((input_height - self.kernel_size) / self.stride) + 1)
		W_new = int(((input_height - self.kernel_size) / self.stride) + 1)
    
		out = np.zeros((num_image, C, H_new, W_new))

		# (100,1,28,28) > (100,1,14,2,14,2) > (100,1,14,14)
		# https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy/49317610
		output = self.input.reshape(num_image, C, input_height//self.kernel_size, self.kernel_size, 
										input_width//self.kernel_size, self.kernel_size).max(axis=(3,5))

		maxs = output.repeat(2, axis=2).repeat(2, axis=3)
		x_window = self.input[:, :, :H_new * self.stride, :W_new * self.stride]
		self.mask = np.equal(x_window, maxs).astype(int)
		
		return output
	    ############################################################################

	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the new delta.

		# https://stackoverflow.com/questions/61954727/max-pooling-backpropagation-using-numpy
		delta_prev = delta.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)
		delta_prev = np.multiply(delta_prev, self.mask)
		pad = np.zeros(self.input.shape)
		pad[:, :, :delta_prev.shape[2], :delta_prev.shape[3]] = delta_prev
		return pad

	    ############################################################################
