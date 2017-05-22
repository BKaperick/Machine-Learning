import numpy as np
from neuralnet import *

# 28x28 pixel images are encoded as a length-784 numpy array
# Outputs are a length-10 numpy array corresponding to 10 decimal digits
layer_map = [784,15,10]

# Initialize network
network = Network(layer_map)

# Load data
num_samples = 15
training_labels, training_images, test_labels, test_images = open_data.get_data(num_samples, training=True)


# Initialize weights randomly
weights = [None]
for i,size in enumerate(network.layer_sizes[:-1]):
    next_size = network.layer_sizes[i+1]
    weights.append(np.random.rand(next_size, size))

biases = []
for i,size in enumerate(network.layer_sizes):
    biases.append(np.random.randn(size))

# Test that everything appears to be working
out = network.cost(training_images, weights, biases, training_labels)
print(out)
