import numpy as np
from neuralnet import Network

layer_map = [2,2,2]
L = len(layer_map)

# indexed 2,3,...,L
weights = [np.ones((2,2)) for l in range(L-1)]
assert(len(weights) == L-1)

# indexed 2,3,...,L
biases = [np.zeros(2) for l in range(L-1)]
assert(len(weights) == L-1)
network = Network(layer_map, weights, biases)

train_inputs = np.asarray(
        [[1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5]]
        )
train_labels = np.asarray(
        [[1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1]]
        )

test_inputs = np.asarray(
        [[1.5, 2.5, 3.5, 4.5, 5.5],
         [1.5, 2.5, 3.5, 4.5, 5.5]]
        )
test_labels = np.asarray(
        [[1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1]]
        )

epochs = 1
eta = 1.0
batch_size = 5
plotting = False

# Train network weights and labels on training data
network.stochastic_gradient_descent(train_inputs, train_labels, batch_size, epochs, eta, test_inputs, test_labels, plot=plotting)
print(network.cost(test_inputs, test_labels))


