import numpy as np
from functions import Function
from tensor import Tensor
import time
from numba import jit
import np_utils
import copy
import pickle
import gzip
from conv_functions import *
from math import ceil,sqrt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

class Dataset:
    def __init__(self):
        self.X = np.concatenate((np.load('../cats2.npy'), np.load('../dogs2.npy')),axis=0)
        self.y = np.concatenate((np.zeros((10000,1)),np.ones((10000,1))),axis=0)
        self.X,self.y = shuffle(self.X,self.y)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # print (y_train.shape)
        # print (X_train.shape)
        # print (y_valid.shape)
        # print (X_valid.shape)

    def get_random_example(self,random_state=0):
        random.seed(random_state)
        return self.X[random.randint(0,len(X)-1)]

class ConvNetwork:
    def __init__(self,k1=None,k2=None,f1=None,f2=None):
        """
        Initialize convolutional and fully connected weights here.
        randn, not rand!!
        Layer weight initializations to random normal kaiming initialization
        unless layer values specified
        """
        if (k1 == None):
            self.k1 = Tensor(np.random.randn(3,3,12,12) * (sqrt(2./10000.)))
        else:
            self.k1 = Tensor(k1)

        if (k2 == None):
            self.k2 = Tensor(np.random.randn(3,3,6,6) * (sqrt(2./7921.)))
        else:
            self.k2 = Tensor(k2)

        if (f1 == None):
            self.f1 = Tensor(np.random.randn(21168,100) * (sqrt(2./21168.)))
        else:
            self.f1 = Tensor(f1)

        if (f2 == None):
            self.f2 = Tensor(np.random.randn(100,1)   * (sqrt(2./100.)))
        else:
            self.f2 = Tensor(f2)

    def forward(self, input_in):
        """
        Forward pass of the network.
        :param input_in: Input tensor of size (batch_size x in_channels x height x width).
        :return: output tensor (batch_size x out_size).
        """
        # a1 = (batch_size x 12 x 89 x 89)
        self.a1 = input_in.conv2d(self.k1).leakyrelu()
        # a2 = (batch_size x 12 x 84 x 84)
        self.a2 = self.a1.conv2d(self.k2).leakyrelu()
        # a3 = (batch_size x 100)
        self.a3 = self.a2.reshape((input_in.value.shape[0],-1)).dot(self.f1).leakyrelu()
        # output = (batch_size x 1)
        self.a4 = self.a3.dot(self.f2)
        self.a5 = self.a4.sigmoid()

        return self.a5

    def update(self, lr):
        """
        Update the weights of the network using SGD.
        """
        self.k1.value += -lr * self.k1.grad
        self.k2.value += -lr * self.k2.grad
        self.f1.value += -lr * self.f1.grad
        self.f2.value += -lr * self.f2.grad

    def zero_grad(self):
        """
        Reset gradients for the weights of the network.
        """
        self.k1.zero_grad()
        self.k2.zero_grad()
        self.f1.zero_grad()
        self.f2.zero_grad()

    # function for thresholding from sigmoid to predicted class
    def predict(self,X):
        # turn x into tensor
        X_valid_tensor = Tensor(X.reshape((-1,3,100,100)))
        predictions = self.forward(X_valid_tensor).value
        greater_than_threshold = predictions > 0.5
        less_than_threshold = predictions <= 0.5
        predictions[greater_than_threshold] = 1
        predictions[less_than_threshold] = 0
        return predictions

    def save_weights(self,epoch):
        np.save("k1_" + str(epoch) + ".npy",self.k1.value)
        np.save("k2_" + str(epoch) + ".npy",self.k2.value)
        np.save("f1_" + str(epoch) + ".npy",self.f1.value)
        np.save("f2_" + str(epoch) + ".npy",self.f2.value)

    def test(self, X, y):
        """
        Test the model
        :param model: Trained ConvNetwork object.
        :param X: Validation set inputs.
        :param y: Validation set targets.
        :return: Accuracy of model.
        """
        # make a one hot vector from the indices
        valid_one_hot = self.predict(X)

        # count incorrect predictions by subtracting, squaring, summing predictions and counting non-zeros
        num_incorrect = np.count_nonzero(np.sum(np.square(valid_one_hot - y),axis=1))

        # calculate accuracy
        accuracy = (len(y) - num_incorrect)/len(y)

        return accuracy

    def train(self, X, y, X_valid, y_valid, epochs=5, batch_size=10,lr=0.001):
        """
        Train the model and validate it every few epochs.
        :param X: Training set inputs.
        :param y: Training set targets.
        :param X_valid: Validation set inputs.
        :param y_valid: Validation set targets.
        :return: Training error for every epoch.
        :return: Validation accuracy every few epochs.
        """
        # For each minibatch in each epoch,
        # 1. do a forward pass on the minibatch
        # 2. Compute Loss on minibatch
        # 3. Do a backward pass to get gradients
        # 4. Update parameters
        # 5. Compute error over epoch

        # compute batch size:
        num_batches = ceil(len(X)/batch_size)

        # store the loss and validation accuracy for each epoch
        loss = []
        validation_accuracy = []

        # for each epoch: forward, loss compute, backward, update, store loss
        for epoch in range(0,epochs):
            X,y = shuffle(X,y)
            print ("weights ", np.round(np.mean(np.abs(self.k1.value)),decimals=8), " ", np.round(np.mean(np.abs(self.k2.value)),decimals=8), " ", np.round(np.mean(np.abs(self.f1.value)),decimals=8), " ", np.round(np.mean(np.abs(self.f2.value)),decimals=8))
            for batch in range(0,num_batches):
                # get upper and lower indices on batch
                lower = batch*batch_size
                upper = min((batch+1)*batch_size,len(X))
                batch_tensor_x = Tensor(X[lower:upper].reshape((-1,3,100,100)))
                batch_tensor_y = Tensor(y[lower:upper])
                # get forward prediction, y_hat (tensor)
                predictions = self.forward(batch_tensor_x)
                # error is the mean squared difference between prediction and ground truth
                MSE = ((predictions - batch_tensor_y).pow(2)).mean()
                # perform back prop using AD
                MSE.backward()
                # update the model
                self.update(lr)
                print ("grad ", np.round(np.mean(np.abs(self.k1.grad)),decimals=8), " ", np.round(np.mean(np.abs(self.k2.grad)),decimals=8), " ", np.round(np.mean(np.abs(self.f1.grad)),decimals=8), " ", np.round(np.mean(np.abs(self.f2.grad)),decimals=8))
                # zero out the gradients before backprop each time
                self.zero_grad()

            # store MSE
            X_tensor = Tensor(X.reshape((-1,3,100,100)))
            training_error = np.mean(np.square(self.forward(X_tensor).value - y))
            loss.append(training_error)
            print ("layers ", np.round(np.mean(np.abs(self.a1.value)),decimals=8), " ", np.round(np.mean(np.abs(self.a2.value)),decimals=8), " ", np.round(np.mean(np.abs(self.a3.value)),decimals=8), " ", np.round(np.mean(np.abs(self.a4.value)),decimals=8))
            print ("weights ", np.round(np.mean(np.abs(self.k1.value)),decimals=8), " ", np.round(np.mean(np.abs(self.k2.value)),decimals=8), " ", np.round(np.mean(np.abs(self.f1.value)),decimals=8), " ", np.round(np.mean(np.abs(self.f2.value)),decimals=8))
            # print ("predictions ", model.forward(X_tensor).value.reshape((10)), " ", y.reshape((10)))
            # store validation accuracy
            validation_accuracy.append(self.test(X_valid,y_valid))
            self.zero_grad()

        return loss, validation_accuracy

    @staticmethod
    @jit(nopython=True)
    def shap_pixel_helper(example,dataset,num_iterations):
        height = example.shape[0]
        width = example.shape[1]
        channels = example.shape[2]
        shap_map = np.zeros((height,width,1))
        for iteration in range(num_iterations):
            Z = dataset.get_random_example()
            # i: height, j:width, assuming shape is (height,width,channel)
            for i in range(height):
                for j in range(width):
                    mask = np.random.rand(height,width,1)
                    mask = np.tile(mask,3)
                    feature_permutation = np.zeros((height,width,channels))
                    feature_permutation[mask > 0.5] = Z[mask > 0.5]
                    feature_permutation[mask <= 0.5] = example[mask <= 0.5]
                    # with feature i,j from example
                    feature_permutation[i][j] = example[i][j]
                    phi_x = self.forward(feature_permutation)
                    # without feature i,j (feature comes from Z)
                    feature_permutation[i][j] = Z[i][j]
                    phi_z = self.forward(feature_permutation)
                    shap_map[i][j] += phi_x - phi_z

        return shap_map/num_iterations

    def shap_pixel(self,example,dataset,num_iterations=10):
        return shap_pixel_helper(example,dataset,num_iterations)
