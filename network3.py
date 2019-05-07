import numpy as np
from functions import Function
from tensor import Tensor
import time
import np_utils
import copy
import pickle
import gzip
from conv_functions import *
from math import ceil,sqrt,inf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
from sklearn.cluster import KMeans


class Dataset:
    def __init__(self,dataset_choice="mnist"):
        if (dataset_choice=="mnist"):
            self.X_train, self.y_train, self.X_valid, self.y_valid = pickle.load(gzip.open("data/mnist21x21_3789_one_hot.pklz", "rb"))
            print ("MNIST DATASET LOADED")
        elif (dataset_choice=="catsdogs"):
            X = np.concatenate((np.load('../cats2.npy'), np.load('../dogs2.npy')),axis=0)
            y = np.concatenate((np.zeros((10000,1)),np.ones((10000,1))),axis=0)
            X,y = shuffle(X,y)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            print ("CATS AND DOGS DATASET LOADED")
        else:
            raise ValueError("Must specify a dataset: Dataset('mnist') or Dataset('catsdogs')")

    def get_random_example(self):
        return self.X_train[random.randint(0,len(self.X_train)-1)]

class ConvNetwork:
    def __init__(self,dataset_choice="mnist",k1=None,k2=None,f1=None,f2=None):
        """
        Initialize convolutional and fully connected weights here.
        Layer weight initializations to random normal kaiming initialization
        unless layer values specified.
        Mnist is default convolutional network configuration if weights are not
        specified.
        """
        self.dataset_choice = dataset_choice
        if (dataset_choice=="mnist" and k1 is None):
            self.k1 = Tensor(np.random.randn(2,1,6,6) * (sqrt(2.)/sqrt(441.)))
        elif (dataset_choice=="catsdogs" and k1 is None):
            self.k1 = Tensor(np.random.randn(3,3,12,12) * (sqrt(2./10000.)))
        elif (dataset_choice=="mnist" or dataset_choice=="catsdogs"):
            self.k1 = Tensor(k1)
        else:
            raise ValueError("ERROR: INVALID DATASET SPECIFIED \n Please specify either 'mnist or 'catsdogs'")

        if (dataset_choice=="mnist" and k2 is None):
            self.k2 = Tensor(np.random.randn(8,2,6,6) * (sqrt(2.)/sqrt(256.)))
        elif (dataset_choice=="catsdogs" and k2 is None):
            self.k2 = Tensor(np.random.randn(3,3,6,6) * (sqrt(2./7921.)))
        elif (dataset_choice=="mnist" or dataset_choice=="catsdogs"):
            self.k2 = Tensor(k2)
        else:
            raise ValueError("ERROR: INVALID DATASET SPECIFIED \n Please specify either 'mnist or 'catsdogs'")

        if (dataset_choice=="mnist" and f1 is None):
            self.f1 = Tensor(np.random.randn(968,100) * (sqrt(2.)/sqrt(968.)))
        elif (dataset_choice=="catsdogs" and f1 is None):
            self.f1 = Tensor(np.random.randn(21168,100) * (sqrt(2./21168.)))
        elif (dataset_choice=="mnist" or dataset_choice=="catsdogs"):
            self.f1 = Tensor(f1)
        else:
            raise ValueError("ERROR: INVALID DATASET SPECIFIED \n Please specify either 'mnist or 'catsdogs'")

        if (dataset_choice=="mnist" and f2 is None):
            self.f2 = Tensor(np.random.randn(100,4)   * (sqrt(2.)/sqrt(100.)))
        elif (dataset_choice=="catsdogs" and f2 is None):
            self.f2 = Tensor(np.random.randn(100,1)   * (sqrt(2.)/sqrt(100.)))
        elif (dataset_choice=="mnist" or dataset_choice=="catsdogs"):
            self.f2 = Tensor(f2)
        else:
            raise ValueError("ERROR: INVALID DATASET SPECIFIED \n Please specify either 'mnist or 'catsdogs'")

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
        if (self.dataset_choice=="mnist"):
            X_valid_tensor = Tensor(X.reshape((-1,1,21,21)))
            predictions = self.forward(X_valid_tensor).value
            prediction_one_hot = np.zeros(predictions.shape)
            valid_max_index = np.argmax(predictions,axis=1)
            prediction_one_hot[np.arange(len(valid_max_index)),valid_max_index] = 1
            return prediction_one_hot
        else:
            X_valid_tensor = Tensor(X.reshape((-1,3,100,100)))
            predictions = self.forward(X_valid_tensor).value
            greater_than_threshold = predictions > 0.5
            less_than_threshold = predictions <= 0.5
            predictions[greater_than_threshold] = 1
            predictions[less_than_threshold] = 0
            return predictions

    def save_weights(self,epoch):
        np.save("k1_" + str(epoch) + "_" + self.dataset_choice + ".npy",self.k1.value)
        np.save("k2_" + str(epoch) + "_" + self.dataset_choice + ".npy",self.k2.value)
        np.save("f1_" + str(epoch) + "_" + self.dataset_choice + ".npy",self.f1.value)
        np.save("f2_" + str(epoch) + "_" + self.dataset_choice + ".npy",self.f2.value)

    def test(self, X, y):
        """
        Test the model
        :param X: Validation set inputs.
        :param y: Validation set targets.
        :return: Accuracy of model.
        """
        # get predictions
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
            for batch in range(0,num_batches):
                # get upper and lower indices on batch
                lower = batch*batch_size
                upper = min((batch+1)*batch_size,len(X))
                if (self.dataset_choice=="mnist"):
                    batch_tensor_x = Tensor(X[lower:upper].reshape((-1,1,21,21)))
                else:
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
                # zero out the gradients before backprop each time
                self.zero_grad()

            # store MSE
            if (self.dataset_choice=="mnist"):
                X_tensor = Tensor(X.reshape((-1,1,21,21)))
            else:
                X_tensor = Tensor(X.reshape((-1,3,100,100)))
            training_error = np.mean(np.square(self.forward(X_tensor).value - y))
            loss.append(training_error)

            # store validation accuracy
            epoch_accuracy = self.test(X_valid,y_valid)
            validation_accuracy.append(epoch_accuracy)
            self.zero_grad()
            # self.save_weights(epoch)
            print (epoch, " VALIDATION ACCURACY: ", np.round(epoch_accuracy,decimals=4))

        return loss, validation_accuracy

    def normalize_shap(self,image):
        if (np.max(image) != 0 and np.min(image) != 0):
            image[image < 0] = (image[image < 0] - np.min(image[image < 0]))/(np.max(image[image < 0])-np.min(image[image < 0])) - 1
            image[image > 0] = (image[image > 0] - np.min(image[image > 0]))/(np.max(image[image > 0])-np.min(image[image > 0]))

        return image

    def shap_pixel(self,example,label,dataset,num_iterations=10):
        """
        Approximate Shapley values for each individual pixel.
        :param example: the image to calculate Shapley values for
        :param label: the class label assigned to the image
        :param dataset: a dataset object with the rest of the examples
        :param num_iterations: number of iterations used to approximate Shapley Values
        :return: a map the same height and width as the image with a Shapley value
                    for each pixel
        """

        if (self.dataset_choice=="mnist"):
            example = example.reshape((21,21,1))
        else:
            example = example.reshape((100,100,3))

        height = example.shape[0]
        width = example.shape[1]
        channels = example.shape[2]
        shap_map = np.zeros((height,width,1))

        for iteration in range(num_iterations):
            Z = dataset.get_random_example().reshape((height,width,channels))

            # i: height, j:width, assuming shape is (height,width,channel)
            for i in range(height):
                for j in range(width):
                    mask = np.random.rand(height,width,1)
                    if (self.dataset_choice=="catsdogs"):
                        mask = np.tile(mask,3)
                    feature_permutation = np.zeros((height,width,channels))
                    feature_permutation[mask > 0.5] = Z[mask > 0.5]
                    feature_permutation[mask <= 0.5] = example[mask <= 0.5]

                    # with feature i,j from example
                    feature_permutation[i][j] = example[i][j]
                    phi_x = self.forward(Tensor(feature_permutation.reshape((1,channels,height,width)))).value

                    # without feature i,j (feature comes from Z)
                    feature_permutation[i][j] = Z[i][j]
                    phi_z = self.forward(Tensor(feature_permutation.reshape((1,channels,height,width)))).value

                    # because the mnist dataset classifies into 4 classes, we multiply
                    # the phi value by the one hot vector to retain only the contibution
                    # to the correct class
                    if (self.dataset_choice=="mnist"):
                        shap_map[i][j] += np.sum(np.multiply(phi_x,label)) - np.sum(np.multiply(phi_z,label))
                    else:
                        if (label==1):
                            shap_map[i][j] += np.asscalar(phi_x) - np.asscalar(phi_z)
                        else:
                            shap_map[i][j] += np.asscalar(phi_z) - np.asscalar(phi_x)

        return (self.normalize_shap(shap_map/num_iterations))

    def kmean_image(self,image,num_clusters):
        """
        Cluster an image using kmeans into num_clusters
        :param image: image to be clustered (width,height,channels)
        :param num_clusters: number of clusters to segment image into.
        :return: cluster map of the image with the same height and width, but only
                    one channel containing the cluster index the corresponding
                    pixel belongs to
        """

        model = KMeans(n_clusters=num_clusters, random_state=0)
        test_image = np.array(image.reshape((image.shape[0]*image.shape[1],-1)))
        cluster_map = np.zeros((image.shape[0]*image.shape[1]))
        model.fit(test_image)

        for i,pixel in enumerate(test_image):
            minimum_distance = inf
            closest_cluster = None
            closest_cluster_index = None

            for c,cluster in enumerate(model.cluster_centers_):
                if (np.sqrt(np.sum(np.square(cluster-pixel))) < minimum_distance):
                    minimum_distance = np.sqrt(np.sum(np.square(cluster-pixel)))
                    closest_cluster = cluster
                    closest_cluster_index = c

            test_image[i] = np.round(closest_cluster)
            cluster_map[i] = closest_cluster_index

        return cluster_map.reshape(image.shape[0:2])

    def shap_cluster(self,example,label,dataset,num_iterations=10,num_clusters=3):
        """
        Approximate Shapley values for each cluster in a clustered image.
        :param exmaple: image we want Shapley values for.
        :param label: class label of image.
        :param dataset: dataset object with rest of training examples.
        :param num_iterations: number of iterations used to approximate Shapley value.
        :param num_clusters: number of clusters to segment image into.
        :return: shap map with the same height and width as example image, but only
                one channel containing the shapley value for the cluster that the
                corresponding pixel is a part of.
        """

        if (self.dataset_choice=="mnist"):
            example = example.reshape((21,21,1))
        else:
            example = example.reshape((100,100,3))

        height = example.shape[0]
        width = example.shape[1]
        channels = example.shape[2]
        shap_map = np.zeros((height,width,1))
        example_clustered = self.kmean_image(example,num_clusters).reshape(height,width,1)
        if (self.dataset_choice=="catsdogs"):
            example_clustered = np.tile(example_clustered,3)
        for iteration in range(num_iterations):
            Z = dataset.get_random_example().reshape((height,width,channels))

            # i: height, j:width, assuming shape is (height,width,channel)
            for cluster_of_interest in range(num_clusters):
                    a = random.randint(0,2**num_clusters-1)
                    mask = "0" * (num_clusters - len(bin(a)[2:])) + bin(a)[2:]
                    feature_permutation = np.zeros((height,width,channels))

                    for c in range(num_clusters):
                        if (mask[c] == "1"):
                            feature_permutation[example_clustered == c] = Z[example_clustered == c]
                        else:
                            feature_permutation[example_clustered == c] = example[example_clustered == c]

                    feature_permutation[example_clustered == cluster_of_interest] = example[example_clustered == cluster_of_interest]
                    phi_x = self.forward(Tensor(feature_permutation.reshape((1,channels,height,width)))).value
                    # without feature i,j (feature comes from Z)
                    feature_permutation[example_clustered == cluster_of_interest] = Z[example_clustered == cluster_of_interest]
                    phi_z = self.forward(Tensor(feature_permutation.reshape((-1,channels,height,width)))).value
                    if (self.dataset_choice=="mnist"):
                        shap_map[(example_clustered == cluster_of_interest)[:,:,0]] += np.sum(np.multiply(phi_x,label)) - np.sum(np.multiply(phi_z,label))
                    else:
                        if (label==1):
                            shap_map[(example_clustered == cluster_of_interest)[:,:,0]] += np.asscalar(phi_x) - np.asscalar(phi_z)
                        else:
                            shap_map[(example_clustered == cluster_of_interest)[:,:,0]] += np.asscalar(phi_z) - np.asscalar(phi_x)

        return self.normalize_shap(shap_map/num_iterations)

    def grid_image(self,image,height,width):
        grid_height = ceil(height/10)
        grid_width = ceil(width/10)

        grid_map = np.zeros((height,width))
        superpixel = 0
        for i in range(10):
            for j in range(10):
                grid_map[i*grid_height:(i+1)*grid_height,j*grid_width:(j+1)*grid_width] = superpixel
                superpixel+=1

        return grid_map,superpixel

    def shap_superpixel(self,example,label,dataset,num_iterations=10):
        """
        Approximate Shapley values for each cluster in a clustered image.
        :param exmaple: image we want Shapley values for.
        :param label: class label of image.
        :param dataset: dataset object with rest of training examples.
        :param num_iterations: number of iterations used to approximate Shapley value.
        :param num_clusters: number of clusters to segment image into.
        :return: shap map with the same height and width as example image, but only
                one channel containing the shapley value for the cluster that the
                corresponding pixel is a part of.
        """

        if (self.dataset_choice=="mnist"):
            example = example.reshape((21,21,1))
        else:
            example = example.reshape((100,100,3))

        height = example.shape[0]
        width = example.shape[1]
        channels = example.shape[2]
        shap_map = np.zeros((height,width,1))
        example_clustered,superpixels = self.grid_image(example,height,width)
        example_clustered = example_clustered.reshape(height,width,1)
        if (self.dataset_choice=="catsdogs"):
            example_clustered = np.tile(example_clustered,3)
        for iteration in range(num_iterations):
            Z = dataset.get_random_example().reshape((height,width,channels))

            # i: height, j:width, assuming shape is (height,width,channel)
            for cluster_of_interest in range(superpixels):
                    a = random.randint(0,2**superpixels-1)
                    mask = "0" * (superpixels - len(bin(a)[2:])) + bin(a)[2:]
                    feature_permutation = np.zeros((height,width,channels))

                    for c in range(superpixels):
                        if (mask[c] == "1"):
                            feature_permutation[example_clustered == c] = Z[example_clustered == c]
                        else:
                            feature_permutation[example_clustered == c] = example[example_clustered == c]

                    feature_permutation[example_clustered == cluster_of_interest] = example[example_clustered == cluster_of_interest]
                    phi_x = self.forward(Tensor(feature_permutation.reshape((1,channels,height,width)))).value
                    # without feature i,j (feature comes from Z)
                    feature_permutation[example_clustered == cluster_of_interest] = Z[example_clustered == cluster_of_interest]
                    phi_z = self.forward(Tensor(feature_permutation.reshape((-1,channels,height,width)))).value
                    if (self.dataset_choice=="mnist"):
                        shap_map[(example_clustered == cluster_of_interest)[:,:,0]] += np.sum(np.multiply(phi_x,label)) - np.sum(np.multiply(phi_z,label))
                    else:
                        if (label==1):
                            shap_map[(example_clustered == cluster_of_interest)[:,:,0]] += np.asscalar(phi_x) - np.asscalar(phi_z)
                        else:
                            shap_map[(example_clustered == cluster_of_interest)[:,:,0]] += np.asscalar(phi_z) - np.asscalar(phi_x)

        return self.normalize_shap(shap_map/num_iterations)
