from network2 import *

k1 = np.load("k1_4_mnist.npy")
k2 = np.load("k2_4_mnist.npy")
f1 = np.load("f1_4_mnist.npy")
f2 = np.load("f2_4_mnist.npy")

# Train and validate the network
network = ConvNetwork(k1,k2,f1,f2)
dataset = Dataset()
print (dataset.X_valid.shape)
num_train = 4000
num_valid = 1000
epochs = 5
batch_size = 10

_started_at = time.time()
loss,acc = network.train(dataset.X_train[:num_train], dataset.y_train[:num_train], dataset.X_valid[:num_valid], dataset.y_valid[:num_valid], epochs, batch_size)
print (loss)
print (acc)
# example = dataset.X[1]
# np.save("example_cluster2.npy",example)
# np.save("test_shap_cluster2.npy",network.shap_cluster(example,dataset,10))
elapsed = time.time() - _started_at
print('({}s)'.format(round(elapsed, 2)))
