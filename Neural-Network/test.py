import numpy as np
from neuralnet import *

# 28x28 pixel images are encoded as a length-784 numpy array
# Outputs are a length-10 numpy array corresponding to 10 decimal digits
layer_map = [784,30,10]


# Load data
num_samples = 60000
training_labels, training_images, test_labels, test_images = open_data.get_data(num_samples, training=True, test=False)

training_labels = training_labels[:50000]
training_images = training_images[:50000]
test_labels = training_labels[50000:]
test_images = training_images[50000:]


# Initialize weights randomly
weights = []
for i,size in enumerate(layer_map[:-1]):
    next_size = layer_map[i+1]
    weights.append(np.random.rand(next_size, size))

print([np.shape(w) for w in weights])

biases = []
for i,size in enumerate(layer_map[1:]):
    
    biases.append(np.random.randn(size))

# Initialize network
network = Network(layer_map, weights, biases)

def label_to_flag_vec(labels):
    # Reshape output scalars to the decimal encoding
    label_vecs = np.zeros((10,len(labels)))
    for i,val in enumerate(labels):
        label_vecs[int(val),i] = 1
    return label_vecs

training_label_vecs = label_to_flag_vec(training_labels)
test_label_vecs = label_to_flag_vec(test_labels)

epochs = 30
eta = 3
batch_size = 10

# Train network weights and labels on training data
network.stochastic_gradient_descent(training_images, training_label_vecs, batch_size, epochs, eta, test_images, test_label_vecs, tick=1, plot=True)
print(network.cost(test_images, test_label_vecs))

## Test that everything appears to be working
#out = network.cost(training_images, training_label_vecs)
#print(out)
