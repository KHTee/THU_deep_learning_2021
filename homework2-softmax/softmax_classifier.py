import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here

    loss = 0.0
    num_train = input.shape[0]
    num_classes = W.shape[1]

    score = np.dot(input, W) # (N,C)
    prediction = np.argmax(score, axis=1)
    score -= np.max(score, axis=1, keepdims=True)

    # # cross entropy loss
    # # take exponent of the score and normalized with sum of all exponents.
    probs = np.exp(score) # (N,C)
    e_y = np.sum(np.multiply(probs,label), axis=1) # (N,) probability for correct class
    e_sum = np.sum(probs, axis=1) # (N,) sum of probability over all classes

    # implementation of loss equivalent l_i = -f_y_i + log sum_j(e^(f_j))
    # loss = np.sum(-np.log(e_y/e_sum)) # sum of -log across all samples.
    # loss /= num_train # average loss
    loss = np.sum(-1 * e_y) + np.sum(np.log(e_sum))
    loss /= num_train

    loss += lamda * np.sum(W * W) # regularization 

    # Gradient
    delta_score = probs / e_sum.reshape(num_train,1) # (N,C)
    delta_score -= label # (NxC)
    gradient = np.dot(input.T, delta_score)
    gradient /= num_train
    gradient += lamda * 2 * W

    ############################################################################

    return loss, gradient, prediction
