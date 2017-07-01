import numpy as np
from neuralnet import *

# 28x28 pixel images are encoded as a length-784 numpy array
# Outputs are a length-10 numpy array corresponding to 10 decimal digits
layer_map = [784,30,10]


# Load data
num_samples = 6000
split_at = 5000
test_data = False
training_labels, training_images, test_labels, test_images = open_data.get_data(num_samples, training=True, test=test_data)

#np.random.shuffle(training_labels.T)
#np.random.shuffle(training_images.T)
#if test_data:
#    np.random.shuffle(test_labels.T)
#    np.random.shuffle(test_images.T)

test_labels = training_labels[split_at:]
test_images = (training_images[:,split_at:] / 255) - .5
training_labels = training_labels[:split_at]
training_images = (training_images[:,:split_at] / 255) - .5

# Initialize network
network = Network(layer_map)

def label_to_flag_vec(labels):
    # Reshape output scalars to the decimal encoding
    label_vecs = np.zeros((10,len(labels)))
    for i,val in enumerate(labels):
        label_vecs[int(val),i] = 1.0
    return label_vecs

training_label_vecs = label_to_flag_vec(training_labels)
test_label_vecs = label_to_flag_vec(test_labels)

epochs     = 10000
eta        = 3.0
batch_size = 10
tick       = 50
plotting = True

# Train network weights and labels on training data
network.stochastic_gradient_descent(training_images, training_label_vecs, batch_size, epochs, eta, test_images, test_label_vecs, tick=tick, plot=plotting)
print(network.cost(test_images, test_label_vecs))

## Test that everything appears to be working
#out = network.cost(training_images, training_label_vecs)
#print(out)
