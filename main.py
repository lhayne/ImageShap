from network import *

k1 = np.load("weights/k19.npy")
k2 = np.load("weights/k29.npy")
f1 = np.load("weights/f19.npy")
f2 = np.load("weights/f29.npy")

# Train and validate the network
network = ConvNetwork(k1,k2,f1,f2)
dataset = Dataset()
num_train = 100
num_valid = 400
epochs = 5
batch_size = 10

_started_at = time.time()
# loss,acc = network.train(dataset.X_train[:num_train], dataset.y_train[:num_train], dataset.X_valid[:num_valid], dataset.y_valid[:num_valid], epochs, batch_size)
example = dataset.X[1]
np.save("example_cluster2.npy",example)
np.save("test_shap_cluster2.npy",network.shap_cluster(example,dataset,10))
elapsed = time.time() - _started_at
print('({}s)'.format(round(elapsed, 2)))
