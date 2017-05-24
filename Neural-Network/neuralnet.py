import numpy as np
import open_data
import random

def sigmoid(z):
    ''' Just a logistic function component-wise on z '''
    return 1 / (1 + np.exp(-z));

def sigmoid_deriv(z):
    ''' Recursive formulation of Sigmoid function derivative '''
    return np.exp(-z) * (sigmoid(z)**2)

def cost(output_emp, output_act):
    ''' Mean squared error between empirical and actual outputs '''
    n = np.shape(output_emp)[1]
    cost_val = 0
    print(np.shape(output_emp), np.shape(output_act))
    for i in range(n):
        cost_val += np.linalg.norm(output_emp[:,i] - output_act[:,i])**2

    # Standard normalization factor
    cost_val *= (.5/n)
    return cost_val


    


class Network:
    ''' 
    Feed-Forward Network -
    Stores layers of neurons and their connections
    '''
    def __init__(self, layers, weights, biases):
        '''
        Initialize the network-
        layers - array of layer sizes, 
                with layers[0] as input and layers[-1] as output
        '''
        self.layer_sizes = layers
        self.num_layers = len(layers)
        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        self.init_layers(weights, biases)

    def init_layers(self, weights, bias):
        ''' Initialize the neurons, the layers, and their connections '''
        self.layers = []
        for i,layer in enumerate(self.layer_sizes):
            if i == 0:
                self.layers.append(InputLayer(self.layer_sizes[0]))
            else:
                self.layers.append(Layer(weights[i], bias[i]))
    
    def update_layers(self, weights, bias):
        ''' Update weights and biases in the neuron layers '''
        for i,layer in enumerate(self.layer_sizes):
            self.layers.append(Layer(weights[i], bias[i]))
    
    def run(self, inputs, weights = None, biases = None, log = False):
        '''
        Feed inputs into the network as the input layer values, and return
        the output layer's values
        log - Records each layer's output.  This information is necessary
                when called from self.backpropagate
        '''
        
        # If new parameters are inputted, update layers' attributes
        if weights or biases:
            if self.layers:
                self.update_layers(weights, biases) 
            else:
                self.init_layers(weights, biases) 
        
        if log:
            inputs = [inputs]

        for i, layer in enumerate(self.layers):
            # Inputs for the next layer are precisely the 
            # outputs of this layer
            if log:
                if i > 0:
                    inputs.append(layer.evaluate(inputs[-1]))
            else:
                inputs = layer.evaluate(inputs)

        return inputs
    
    def back_propagate_vectorized(self, inputs, correct_outputs, weights = None, biases = None):
        layer_outputs = self.run(inputs, weights, biases, log = True)
        last_weighted_input = np.matmul(self.layers[-1].weight, layer_outputs[-2]) + self.layers[-1].bias
        delta_lplus1 = (layer_outputs[-1] - correct_outputs) * sigmoid_deriv(last_weighted_input)
        
        weight_grads = [np.outer(delta_lplus1, layer_outputs[-2])]
        bias_grads = [delta_lplus1]
        for l in range(self.num_layers-2,0,-1):
            weighted_input = np.matmul(self.layers[l].weight, layer_outputs[l-1]) + self.layers[l].bias
            delta_l = np.matmul(np.transpose(self.layers[l+1].weight), delta_lplus1) * sigmoid_deriv(weighted_input)
            weight_grads.insert(0, np.outer(delta_l, layer_outputs[l-1]))
            bias_grads.insert(0, delta_l)

            delta_lp1 = delta_l
        return weight_grads, bias_grads

    def back_propagate(self, inputs, correct_outputs, weights = None, biases = None):

        # Array containing a_0, ..., a_{L-1}
        layer_outputs = self.run(inputs, weights, biases, log = True)

        # z_{L-1} = w_{L-1}*a_{L-2} + b_{L-1}
        last_weighted_input = np.matmul(self.layers[-1].weight, layer_outputs[-2]) + self.layers[-1].bias

        # delt_{L-1} = (a_{L-1} - y_{L-1}) * s'(z_{L-1})
        delta_lplus1 = (layer_outputs[-1] - correct_outputs) * sigmoid_deriv(last_weighted_input)

        # dC/dw_{L-1} = delt_{L-1} * a_{L-2}^T
        weight_grads = [np.outer(delta_lplus1, layer_outputs[-2])]

        # dC/db_{L-1} = delt_{L-1}
        bias_grads = [delta_lplus1]

        
        # l = L-2, ..., 1
        for l in range(self.num_layers-2,0,-1):

            # z_{L-2} = w_{L-2} * a_{L-3} + b_{L-2}
            weighted_input = np.matmul(self.layers[l].weight, layer_outputs[l-1]) + self.layers[l].bias

            # delt_{L-2} = w_{L-1}^T * delt_{L-1} * s'(z_{L-2})
            delta_l = np.matmul(np.transpose(self.layers[l+1].weight), delta_lplus1) * sigmoid_deriv(weighted_input)

            # dC/dw_{L-2}
            weight_grads.insert(0, np.outer(delta_l, layer_outputs[l-1]))
            bias_grads.insert(0, delta_l)

            delta_lp1 = delta_l
        return weight_grads, bias_grads
        
    def stochastic_gradient_descent(self, train_inputs, train_outputs, batch_size, epochs, eta, test_inputs = [], test_outputs = None, tick = 100):
        '''
        Performs gradient descent on the given data
        training_data - Array of 2-tuples of the form (input, output)
        epochs - Number of epochs to train for
        eta - Learning rate
        test_data - Evaluates network after each epoch, printing progress
        '''
        for j in range(epochs):
            if j % 50 == 0:
                print("{0} epochs completed".format(j))
            
            weight_grads = [np.zeros(np.shape(l.weight)) for l in self.layers[1:]]
            bias_grads = [np.zeros(np.shape(l.bias)) for l in self.layers[1:]]
            
            # Take only a subset
            indices = random.sample(range(np.shape(train_inputs)[1]), batch_size)
            batch_inputs = train_inputs[:,indices]
            batch_outputs = train_outputs[:,indices]
            

            for sample_index in range(batch_size):
                sample_in = batch_inputs[:,sample_index]
                sample_out = batch_outputs[:,sample_index]
                
                sample_weight_grad, sample_bias_grad = self.back_propagate(sample_in, sample_out)
                weight_grads = [weight_grads[l] + sample_weight_grad[l] for l in range(len(weight_grads))]
                bias_grads = [bias_grads[l] + sample_bias_grad[l] for l in range(len(bias_grads))]
            
            for i, layer in enumerate(self.layers[1:]):
                layer.weight = layer.weight - eta*weight_grads[i]/batch_size
                layer.bias = layer.bias - eta*bias_grads[i]/batch_size
            
            if len(test_inputs) > 0 and j % tick == 0:
                print("Generation {0}: {1}".format(j, self.cost(test_inputs, test_outputs)))

    def cost(self, inputs, correct_outputs, weights = None, biases = None):
        '''
        Given a set of input data and correct outputs,
        compute MSE of the difference in the actual
        and expected outputs -
        inputs - each column is a new input sample
        '''
        n = np.shape(inputs)[1]

            
        outputs = np.empty((self.output_dim, n))
        for i in range(n): 
            
            # Cost increases as a smooth function of changes in w,b
            if i == 0:
                output_val = self.run(inputs[:,i], weights, biases)
            else:
                output_val = self.run(inputs[:,i])
            outputs[:,i] = output_val
        
        # Return MSE
        return cost(outputs, correct_outputs)


class Layer:
    ''' Encodes a layer of Sigmoid neurons with a set of weights and biases '''
    def __init__(self, weight, bias):
        self.length = len(bias)
        self.bias = bias
        self.weight = weight
        self.__name__ = "Layer"

    def evaluate(self, inputs):
        return sigmoid(np.matmul(self.weight, inputs) + self.bias)


class InputLayer(Layer):
    ''' Subclass of Layer whos evaluation is just the identity mapping '''
    def __init__(self, length):
        self.length = length
        self.__name__ = "InputLayer"

    def evaluate(self, inputs):
        return inputs
