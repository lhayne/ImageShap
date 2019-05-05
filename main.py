from network import *

# Train and validate the network
network = ConvNetwork()
dataset = Dataset()
num_train = 100
num_valid = 400
epochs = 5
batch_size = 10

_started_at = time.time()
loss,acc = network.train(dataset.X_train[:num_train], dataset.y_train[:num_train], dataset.X_valid[:num_valid], dataset.y_valid[:num_valid], epochs, batch_size)
elapsed = time.time() - _started_at
print('({}s)'.format(round(elapsed, 2)))
