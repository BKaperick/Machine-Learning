import numpy as np
import open_data

def sigmoid(z):
    ''' Just a logistic function component-wise on z '''
    return 1 / (1 + np.exp(-z));

def cost(output_emp, output_act):
    ''' Mean squared error between empirical and actual outputs '''
    n = len(output_emp)
    cost_val = 0
    for i in range(n):
        cost_val += np.linalg.norm(output_emp[i] - output_act[i])**2

    # Standard normalization factor
    cost_val *= (.5/n)

    return cost_val


class Network:
    ''' 
    Feed-Forward Network -
    Stores layers of neurons and their connections
    '''
    def __init__(self, layers=[]):
        '''
        Initialize the network-
        layers - array of layer sizes, 
                with layers[0] as input and layers[-1] as output
        '''
        self.layer_sizes = layers
        self.layers = []
        self.init_neurons()

    def init_neurons(self):
        ''' Initialize the neurons, the layers, and their connections '''
        for i,layer in enumerate(self.layer_sizes):
            
            # Ordering is irrelevant
            new_layer = [Neuron(self, 0) for _ in range(layer)]
            
            # Update node information
            if i > 0:
                for node in new_layer:
                    node.prev_layer = self.layers[-1]
                for node in self.layers[-1]:
                    node.next_layer = new_layer
            
            # Add new layer to the network
            self.layers.append(new_layer)
    
    def eval_layer(self, layer, inputs, weights):
        ''' Apply inputs and weights to this layer -
        weights - ith column is a numpy array for the weights of the ith node's inputs
        '''
        out_vals = np.empty(len(layer), dtype=float);
        for i,node in enumerate(layer):
            out_vals[i] = node.evaluate(inputs, weights[:,i])
        return out_vals

    def run(self, inputs, weights):
        '''
        Feed inputs into the network as the input layer values, and return
        the output layer's values
        '''
        in_vals = inputs
        for i,layer in enumerate(self.layers):
            # First layer is input layer, so no evaluation is done
            if i == 0:
                continue
            # Inputs for the next layer are precisely the outputs of this layer
            in_vals = self.eval_layer(layer, in_vals, weights[i-1])

        out_vals = in_vals
        return out_vals
                    
    def cost(self, inputs, weights, correct_outputs):
        '''
        Given a set of input data and correct outputs,
        compute MSE of the difference in the actual
        and expected outputs -
        inputs - each column is a new input sample
        '''
        n = np.shape(inputs)[1]
        outputs = np.empty(n)
        for i in range(n): 
            # Currently, cost increases only if the maximum output 
            # is incorrect, due to the interpretation of the outputs
            output_val = self.run(inputs[:,i], weights)
            outputs[i] = np.argmax(output_val)
        return cost(outputs, correct_outputs)
                

class Neuron:
    '''
    Sigmoid neurons which evaluate inputs with logistic function
    '''
    def __init__(self, network, bias):
        self.net = network
        self.bias = bias
        self.prev_layer = set()
        self.next_layer = set()

    def evaluate(self, inputs, weights):
        return sigmoid(np.dot(inputs, weights) + self.bias)

# 28x28 pixel images are encoded as a length-784 numpy array
# Outputs are a length-10 numpy array corresponding to 
# 10 decimal digits
layer_map = [784,15,10]

# Initialize network
network = Network(layer_map)

# Load data
num_samples = 15
training_labels, training_images, test_labels, test_images = open_data.get_data(num_samples, training=True)


# Initialize weights randomly
weights = []
for i,size in enumerate(network.layer_sizes[:-1]):
    next_size = network.layer_sizes[i+1]
    weights.append(np.random.rand(size, next_size))

# Test that everything appears to be working
out = network.cost(training_images, weights, training_labels)
print(out)





