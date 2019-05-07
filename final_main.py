from network3 import *

k1 = np.load("weights/catsdogs/k124.npy")
k2 = np.load("weights/catsdogs/k224.npy")
f1 = np.load("weights/catsdogs/f124.npy")
f2 = np.load("weights/catsdogs/f224.npy")
print (k1.shape)
print (k2.shape)
print (f1.shape)
print (f2.shape)
# Train and validate the network
network = ConvNetwork('catsdogs',k1,k2,f1,f2)
dataset = Dataset('catsdogs')

num_train = 10
num_valid = 100
epochs = 5
batch_size = 10

_started_at = time.time()
print (dataset.y_valid[0])
print (network.predict(dataset.X_valid[0]))
# loss,acc = network.train(dataset.X_train[:num_train], dataset.y_train[:num_train], dataset.X_valid[:num_valid], dataset.y_valid[:num_valid], epochs, batch_size)
# print (loss)
# print (acc)
# example = dataset.X_train[0]
# label = dataset.y_train[0]
# np.save("example_pixel_mnist.npy",example)
# np.save("test_shap_pixel_mnist.npy",network.shap_pixel(example,label,dataset,10))
elapsed = time.time() - _started_at
print('({}s)'.format(round(elapsed, 2)))
