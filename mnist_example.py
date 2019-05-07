from network import *
import time

k1 = np.load("weights/mnist/k1_9_mnist.npy")
k2 = np.load("weights/mnist/k2_9_mnist.npy")
f1 = np.load("weights/mnist/f1_9_mnist.npy")
f2 = np.load("weights/mnist/f2_9_mnist.npy")

# Train and validate the network
network = ConvNetwork('mnist',k1,k2,f1,f2)
dataset = Dataset('mnist')

num_train = 100
num_valid = 100
epochs = 5
batch_size = 10

_started_at = time.time()
loss,acc = network.train(dataset.X_train[:num_train], dataset.y_train[:num_train], dataset.X_valid[:num_valid], dataset.y_valid[:num_valid], epochs, batch_size)
print ("Loss     ",loss)
print ("Accuracy ",acc)
elapsed = time.time() - _started_at
print('({}s)'.format(round(elapsed, 2)))
