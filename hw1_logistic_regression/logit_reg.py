#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logistic Regression
Tsinghua Deep Learning 2021
Homework 1

@author: Tee Kah Hui
@studentID: 2020280402
"""

import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import mnist_data_loader

np.seterr(divide = 'ignore')

def load_data():
    mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)
    train_set = mnist_dataset.train
    test_set = mnist_dataset.test
    print('Training dataset size:' , train_set.num_examples)
    print('Test dataset size:' , test_set.num_examples)
    return train_set, test_set

# convert label to one-hot vector
def one_hot_encoder(y):
    one_hot_vec = np.zeros((len(y), 2))

    for i in range(len(y)):
        j = int((y[i]/3) - 1)
        one_hot_vec[i][j] = 1

    return one_hot_vec

# convert one-hot vector to label
def one_hot_decoder(y_pred):
    result = []
    for i in range(len(y_pred)):
        row = list(y_pred[i])
        ans = (row.index(max(y_pred[i])) + 1) * 3
        result.append(ans)

    return result

# sigmoid function
def sigmoid(scores):
   return 1 / (1 + np.exp(-scores))

def predict(X, w):
    y_pred = sigmoid(np.dot(X, w))
    return y_pred

# loss function
def calculate_loss(y, y_pred):
    m = len(y_pred)
    return -1.0 * (1 / m) * np.sum((y * (np.log(y_pred))) + ((1 - y) * (np.log(1 - y_pred))))

def plot_graph(input_list, plot_name, epoch, batch_size, lr):
    x = [i for i in range(len(input_list))]
    plt.figure(figsize=(10,5))
    plt.xlabel('Iterations')
    plt.ylabel(plot_name)
    plt.title(f"{plot_name} vs Iterations, epoch{epoch}_batchsize{batch_size}_lr{lr}")
    plt.plot(x, input_list, c='g')
    plt.savefig(f"{plot_name}_epoch{epoch}_batchsize{batch_size}_lr{lr}.jpg")
    plt.close
    # plt.show()

def train(train_set, epoch=100, batch_size=4, alpha=0.01):
    max_epoch = epoch
    loss_list = []
    acc_list = []
    num_classes = 2

    # initial weight
    num_inputs = train_set.images.shape[1]
    w = np.random.randn(num_inputs, num_classes) / np.sqrt(num_classes*num_inputs)

    for epoch in range(0, max_epoch):
        iter_per_batch = train_set.num_examples // batch_size
        loss_per_epoch = 0
        for batch_id in range(0, iter_per_batch):
            # get the data of next minibatch (have been shuffled)
            batch = train_set.next_batch(batch_size)
            input, label = batch

            X = input
            y = one_hot_encoder(label)
            m = len(X)

            # prediction
            y_pred = predict(X, w)

            # calculate the loss
            loss = calculate_loss(y, y_pred)
            loss_per_epoch += loss

            # gradient descent
            gradient = (1/m) * np.dot(X.T, (y_pred - y))

            # update weights
            w = w - alpha * gradient

        # loss
        curr_loss = loss_per_epoch/iter_per_batch
        loss_list.append(loss_per_epoch/iter_per_batch)

        # accuracy
        y_pred = predict(train_set.images, w)
        y_pred = one_hot_decoder(y_pred)
        y_true = train_set.labels
        acc = accuracy_score(y_true, y_pred)
        acc_list.append(acc)

        sys.stdout.write(f"\r{epoch+1}/{max_epoch} epochs done  |  loss: {curr_loss}  |  acc: {acc}  |")
        sys.stdout.flush()

    print("\n")
    return w, loss_list, acc_list

def test(test_set, w):
    y_pred = predict(test_set.images, w)
    y_pred = one_hot_decoder(y_pred)
    y_true = test_set.labels
    acc = accuracy_score(y_true, y_pred)
    print(f"testing accuracy: %f "%(round(acc*100,2)) + "%\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic regression')
    parser.add_argument('--epoch', default=100, type=int, help='epoch size')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    args = parser.parse_args()

    print("=== Load data ===")
    train_set, test_set = load_data()

    print("=== Start training ===")
    w, loss_list, acc_list = train(train_set, args.epoch, args.batch_size, args.lr)

    print("=== Testing ===")
    test(test_set, w)

    print("=== Plot graphs ===")
    plot_graph(loss_list, 'loss', args.epoch, args.batch_size, args.lr)
    plot_graph(acc_list, 'accuracy', args.epoch, args.batch_size, args.lr)
    print("Graphs plotted.")