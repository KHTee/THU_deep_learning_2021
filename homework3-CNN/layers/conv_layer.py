# -*- encoding: utf-8 -*-

import numpy as np

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve

class ConvLayer():
	"""
	2D convolutional layer.
	This layer creates a convolution kernel that is convolved with the layer
	input to produce a tensor of outputs.
	Arguments:
		inputs: Integer, the channels number of input.
		filters: Integer, the number of filters in the convolution.
		kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
		pad: Integer, the size of padding area.
		trainable: Boolean, whether this layer is trainable.
	"""
	def __init__(self, inputs,
	             filters,
	             kernel_size,
	             pad,
	             trainable=True):
		self.inputs = inputs
		self.filters = filters
		self.kernel_size = kernel_size
		self.pad = pad
		assert pad < kernel_size, "pad should be less than kernel_size"
		self.trainable = trainable

		self.XavierInit()

		self.grad_W = np.zeros_like(self.W)
		self.grad_b = np.zeros_like(self.b)

	def XavierInit(self):
		raw_std = (2 / (self.inputs + self.filters))**0.5
		init_std = raw_std * (2**0.5)

		self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
		self.b = np.random.normal(0, init_std, (self.filters,))

	def forward(self, Input, **kwargs):
		'''
		forward method: perform convolution operation on the input.
		Agrs:
			Input: A batch of images, shape-(batch_size, channels, height, width)
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.

		# Shape of W = (8, 1, 3, 3)
		# Shape of Input = (100, 1, 28, 28)
		# Shape of input_after_pad = (100, 1, 30, 30)
		
		self.input = Input
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		self.input_after_pad = input_after_pad

		self.stride = 1
		num_image, C, input_height, input_width = self.input.shape
		H_prime = int(((input_height + 2*self.pad - self.kernel_size) // self.stride) + 1)
		W_prime = int(((input_width + 2*self.pad - self.kernel_size) // self.stride) + 1)
		output = np.zeros([num_image, self.filters, H_prime, W_prime])

		# ref: https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd
		i0 = np.tile(np.repeat(np.arange(self.kernel_size), self.kernel_size), C)
		i1 = self.stride * np.repeat(np.arange(H_prime), W_prime)
		j0 = np.tile(np.arange(self.kernel_size), self.kernel_size * C)
		j1 = self.stride * np.tile(np.arange(W_prime), H_prime)
		self.i = i0.reshape(-1, 1) + i1.reshape(1, -1)
		self.j = j0.reshape(-1, 1) + j1.reshape(1, -1)
		self.k = np.repeat(np.arange(C), self.kernel_size * self.kernel_size).reshape(-1, 1)

		padded_input = self.input_after_pad[:, self.k, self.i, self.j]
		self.input_cols = padded_input.transpose(1, 2, 0).reshape(self.kernel_size * self.kernel_size * C, -1)
		W_cols = self.W.reshape(self.filters, -1)

		output = np.matmul(W_cols, self.input_cols) 
		output += self.b[:,np.newaxis]
		output = output.reshape(self.filters, H_prime, W_prime, num_image)
		output = output.transpose(3, 0, 1, 2)
 
		return output
		
		# Implementation using for-loop
		#############################
		# for im_num in range(num_image):
		# 	im = input_after_pad[im_num,:,:,:]

		# 	#im2col
		# 	num_channel, im_height, im_width = im.shape
		# 	new_h = int(((im_height - self.kernel_size) // self.stride) + 1)
		# 	new_w = int(((im_width - self.kernel_size) // self.stride) + 1)
		# 	col = np.zeros([new_h*new_w, num_channel*self.kernel_size*self.kernel_size])

		# 	for h in range(new_h):
		# 		for w in range(new_w):
		# 			patch = im[..., h*self.stride:h*self.stride+self.kernel_size, 
		# 					  w*self.stride:w*self.stride+self.kernel_size]
		# 			col[h*new_w+w, :] = np.reshape(patch, -1)

		# 	filter_col = np.reshape(self.W, (self.filters, -1))
		# 	mul = col.dot(filter_col.T) + self.b
			
		# 	#col2im
		# 	F = mul.shape[1]
		# 	temp_output = np.zeros([F, H_prime, W_prime])
		# 	for i in range(F):
		# 		temp_output[i, :, :] = np.reshape(mul[:, i], (H_prime, W_prime))
			
		# 	output[im_num,:,:,:] = temp_output
	    ############################################################################


	def backward(self, delta):
		'''
		backward method: perform back-propagation operation on weights and biases.
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate self.grad_W, self.grad_b, and return the new delta.
		
		num_image, C, input_height, input_width = self.input.shape
		H_padded, W_padded = input_height + 2 * self.pad, input_width + 2 * self.pad
        
		# gradient w.r.t bias
		self.grad_b = np.sum(delta, axis=(0,2,3))
        
		# gradient w.r.t weight
		delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(self.filters, -1)
		dW = np.matmul(delta_reshaped, self.input_cols.T)
		self.grad_W = dW.reshape(self.W.shape)
        
		W_reshape = self.W.reshape(self.filters, -1)
		dX_cols = np.matmul(W_reshape.T, delta_reshaped)
        
		dx_padded = np.zeros((num_image, C, H_padded, W_padded), dtype=dX_cols.dtype)
        
		cols_reshaped = dX_cols.reshape(C * self.kernel_size * self.kernel_size, -1, num_image).transpose(2, 0, 1)
		np.add.at(dx_padded, (slice(None), self.k, self.i, self.j), cols_reshaped)
        
		if self.pad != 0:
		    dx_padded = dx_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        
		return dx_padded
		
		# Implementation using for-loop
		##################
		# delta_prev = np.zeros_like(self.input)

		# num_image, C, input_height, input_width = self.input.shape
		# H_prime = int(((input_height + 2*self.pad - self.kernel_size) // self.stride) + 1)
		# W_prime = int(((input_width + 2*self.pad - self.kernel_size) // self.stride) + 1)

		# for i in range(num_image):
		# 	im = self.input_after_pad[i, :, :, :]

		# 	#im2col
		# 	num_channel, im_height, im_width = im.shape
		# 	new_h = int(((im_height - self.kernel_size) // self.stride) + 1)
		# 	new_w = int(((im_width - self.kernel_size) // self.stride) + 1)
		# 	col = np.zeros([new_h*new_w, num_channel*self.kernel_size*self.kernel_size])

		# 	for h in range(new_h):
		# 		for w in range(new_w):
		# 			patch = im[..., h*self.stride:h*self.stride+self.kernel_size, 
		# 					  w*self.stride:w*self.stride+self.kernel_size]
		# 			col[h*new_w+w, :] = np.reshape(patch, -1)

		# 	filter_col = np.reshape(self.W, (self.filters, -1)).T

		# 	delta_i = delta[i, :, :, :]
		# 	dbias_sum = np.reshape(delta_i, (self.filters, -1))
		# 	dbias_sum = dbias_sum.T

		# 	# self.grad_b += np.sum(dbias_sum, axis=0)
		# 	self.grad_b = np.sum(delta, axis=(0,2,3))
		# 	dmul = dbias_sum

		# 	dfilter_col = (col.T).dot(dmul)
		# 	dim_col = dmul.dot(filter_col.T)

		# 	#col2imback
		# 	H = (H_prime - 1) * self.stride + self.kernel_size
		# 	W = (W_prime - 1) * self.stride + self.kernel_size
		# 	dx_padded = np.zeros([C, H, W])
		# 	for k in range(H_prime * W_prime):
		# 		row = dim_col[k, :]
		# 		h_start = int((k / H_prime) * self.stride)
		# 		w_start = int((k % W_prime) * self.stride)
		# 		dx_padded[:, h_start:h_start+self.kernel_size,
		# 				  w_start:w_start+self.kernel_size] += np.reshape(row, (C, self.kernel_size, self.kernel_size))

		# 	delta_prev[i, :, :, :] = dx_padded[:, self.pad:input_height+self.pad, self.pad:input_width+self.pad]
		# 	self.grad_W += np.reshape(dfilter_col.T, self.W.shape)
		# return delta_prev

	    ############################################################################
