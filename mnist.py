from network2 import *

k1 = np.load("weights/mnist/k1_9_mnist.npy")
k2 = np.load("weights/mnist/k2_9_mnist.npy")
f1 = np.load("weights/mnist/f1_9_mnist.npy")
f2 = np.load("weights/mnist/f2_9_mnist.npy")

# Train and validate the network
network = ConvNetwork(k1,k2,f1,f2)
dataset = Dataset()
print (dataset.X_valid.shape)
num_train = 4000
num_valid = 1000
epochs = 5
batch_size = 10

_started_at = time.time()
# loss,acc = network.train(dataset.X_train[:num_train], dataset.y_train[:num_train], dataset.X_valid[:num_valid], dataset.y_valid[:num_valid], epochs, batch_size)
# print (loss)
# print (acc)
example = dataset.X_train[0]
label = dataset.y_train[0]
np.save("example_pixel_mnist.npy",example)
np.save("test_shap_pixel_mnist.npy",network.shap_pixel(example,label,dataset,10))
elapsed = time.time() - _started_at
print('({}s)'.format(round(elapsed, 2)))
