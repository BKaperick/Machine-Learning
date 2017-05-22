import numpy as np
import open_data

def sigmoid(z):
    ''' Just a logistic function component-wise on z '''
    return 1 / (1 + np.exp(-z));

def cost(output_emp, output_act):
    ''' Mean squared error between empirical and actual outputs '''
    n = np.shape(output_emp)[1]
    cost_val = 0
    for i in range(n):
        cost_val += np.linalg.norm(output_emp[:,i] - output_act[:,i])**2

    # Standard normalization factor
    cost_val *= (.5/n)
    return cost_val


#def gradientDescent(training_data, epochs, eta, test_data=None):
#    '''
#    Performs gradient descent on the given data
#    training_data - Array of 2-tuples of the form (input, output)
#    epochs - Number of epochs to train for
#    eta - Learning rate
#    test_data - Evaluates network after each epoch, printing progress
#    '''
#    if test_data:
#        n_test = len(test_data)
#    n = len(training_data)
#    for j in range(epochs):
        



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
        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        self.layers = []

    def init_layers(self, weights, bias):
        ''' Initialize the neurons, the layers, and their connections '''
        self.layers = []
        for i,layer in enumerate(self.layer_sizes):
            if i == 0:
                self.layers.append(InputLayer(bias[i]))
            else:
                self.layers.append(Layer(weights[i], bias[i]))
    
    def update_layers(self, weights, bias):
        ''' Update weights and biases in the neuron layers '''
        for i,layer in enumerate(self.layer_sizes):
            self.layers.append(Layer(weights[i], bias[i]))
    
    def run(self, inputs):
        '''
        Feed inputs into the network as the input layer values, and return
        the output layer's values
        '''
        for layer in self.layers:
            
            # Inputs for the next layer are precisely the outputs of this layer
            inputs = layer.evaluate(inputs)

        return inputs
                    
    def cost(self, inputs, weights, biases, correct_outputs):
        '''
        Given a set of input data and correct outputs,
        compute MSE of the difference in the actual
        and expected outputs -
        inputs - each column is a new input sample
        '''
        
        if self.layers:
            self.update_layers(weights, biases) 
        else:
            self.init_layers(weights, biases) 

        n = np.shape(inputs)[1]

        # Reshape output scalars to the decimal encoding
        correct_outputs_reshaped = np.zeros((self.output_dim,n))
        for i,val in enumerate(correct_outputs):
            correct_outputs_reshaped[int(val),i] = 1
            
        outputs = np.empty((self.output_dim, n))
        for i in range(n): 
            
            # Cost increases as a smooth function of changes in w,b
            output_val = self.run(inputs[:,i])
            outputs[:,i] = output_val
        
        # Return MSE
        return cost(outputs, correct_outputs_reshaped)
                
class Layer:
    ''' Encodes a layer of Sigmoid neurons with a set of weights and biases '''
    def __init__(self, weights, bias):
        self.length = len(bias)
        self.bias = bias
        self.weights = weights

    def evaluate(self, inputs):
        return sigmoid(np.matmul(self.weights, inputs) + self.bias)


class InputLayer(Layer):
    ''' Subclass of Layer whos evaluation is just the identity mapping '''
    def __init__(self, bias):
        self.length = len(bias)
        self.bias = bias

    def evaluate(self, inputs):
        return inputs


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
