import numpy as np
np.set_printoptions(linewidth=160)
import matplotlib.pyplot as plt
import open_data
import random

from cost_metrics import *

def sigmoid(z):
    ''' 
    Just a logistic function component-wise on z 
    Note: function underflows at z > 745
    '''

    return 1.0 / (1.0 + np.exp(-z));

def sigmoid_deriv(z):
    ''' Recursive formulation of Sigmoid function derivative '''
    sigz = sigmoid(z)
    return sigz * (1-sigz)
   

class Network:
    ''' 
    Feed-Forward Network -
    Stores layers of neurons and their connections
    '''
    def __init__(self, layers):
        '''
        Initialize the network-
        layers - array of layer sizes, 
                with layers[0] as input and layers[-1] as output
        '''
        
        assert(len(layers) > 1)
        self.layers = layers
        self.num_layers = len(layers)
        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        
        # Initialize weights and biases randomly
        self.weights = []
        for i,size in enumerate(layers[:-1]):
            next_size = layers[i+1]
            self.weights.append(np.random.randn(next_size, size))
        self.biases = []
        for i,size in enumerate(layers[1:]):
            self.biases.append(np.random.randn(size))
        
        assert(all([self.biases[i].shape[0] == self.layers[i+1] for i in range(self.num_layers - 1)]))
        assert(all([self.weights[i].shape == (self.layers[i+1],self.layers[i]) for i in range(0,self.num_layers-1)]))

        self.cost_metric = MeanSquaredError
    
    def update_layers(self, weights, bias):
        ''' Update weights and biases in the neuron layers '''
        self.weights = weights
        self.biases = bias
    
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
            # Need to save each state layer output during training
            inputs = [inputs]
       
        for i in range(self.num_layers - 1):
            # Inputs for the next layer are precisely the 
            # outputs of this layer
            if log:
                #if i > 0:
                inputs.append(self.eval_layer(i, inputs[-1]))
            else:
                inputs = self.eval_layer(i, inputs)

        return inputs
    
    def back_propagate(self, inputs, correct_outputs, weights = None, biases = None):

        # Array containing a_0, ..., a_{L-1}
        layer_outputs = self.run(inputs, weights, biases, log = True)

        assert(len(layer_outputs) == self.num_layers)
        assert(all([layer_outputs[i].shape[0] == self.layers[i] for i in range(self.num_layers)]))

        # z_{L-1} = w_{L-1}*a_{L-2} + b_{L-1}
        last_weighted_input = np.matmul(self.weights[-1], layer_outputs[-2]) + self.biases[-1][:,np.newaxis]
        zs = [last_weighted_input.shape]

        # delt_{L-1} = (a_{L-1} - y_{L-1}) * s'(z_{L-1})
        delta_lplus1 = self.cost_metric.gradient(layer_outputs[-1], correct_outputs) * sigmoid_deriv(last_weighted_input)

        # dC/dw_{L-1} = delt_{L-1} * a_{L-2}^T
        weight_grads = [np.matmul(delta_lplus1, layer_outputs[-2].T)]

        # dC/db_{L-1} = delt_{L-1}
        bias_grads = [np.sum(delta_lplus1,1)]

        # l = L-2, ..., 1
        for l in range(self.num_layers-2,0,-1):

            # z_{L-2} = w_{L-2} * a_{L-3} + b_{L-2}
            # w[self.num_layers-2], layer_outputs[self.num_layers-2] + 
            weighted_input = np.matmul(self.weights[l-1], layer_outputs[l-1]) + self.biases[l-1][:,np.newaxis]
            zs.insert(0,weighted_input.shape)

            # delt_{L-2} = w_{L-1}^T * delt_{L-1} * s'(z_{L-2})
            # delt_1 = w_2^T * delt_2 * s'(z_0)
            delta_l = np.matmul(np.transpose(self.weights[l]), delta_lplus1) 
            
            # Element-wise multiplication by s'(z_{L-2})
            delta_l *= sigmoid_deriv(weighted_input)

            # dC/dw_{L-2} = delta_{L-2} * (a_{L-3})^T
            weight_grads.insert(0, np.matmul(delta_l, layer_outputs[l-1].T))

            # Sum together each sample's delta_l
            bias_grads.insert(0, np.sum(delta_l,1))

            delta_lplus1 = delta_l
        
        return weight_grads, bias_grads
        
    def stochastic_gradient_descent(self, train_inputs, train_outputs, batch_size, epochs, eta, test_inputs = [], test_outputs = None, tick = 10, verbose=False):
        '''
        Performs gradient descent on the given data
        training_data - Array of 2-tuples of the form (input, output)
        epochs - Number of epochs to train for
        eta - Learning rate
        test_data - Evaluates network after each epoch, printing progress
        '''
        cost_points = []
        for j in range(epochs):
            if j % tick == 0 and verbose:
                print("{0} epochs completed".format(j))
            
            # Random ordering if indices up to number of input samples
            shuffled_indices = np.random.choice(train_inputs.shape[1], train_inputs.shape[1])
            for m in range(int(train_inputs.shape[1] / batch_size)):
                indices = shuffled_indices[m:m + batch_size]
                batch_inputs = train_inputs[:,indices]
                batch_outputs = train_outputs[:,indices]
                
                weight_grads, bias_grads = self.back_propagate(batch_inputs, batch_outputs)
                scaling = eta / batch_size
                
                for i,layer in enumerate(self.layers[1:]):
                    self.weights[i] -= scaling * weight_grads[i]
                    self.biases[i]  -= scaling * bias_grads[i]
            
            if len(test_inputs) > 0 and j % tick == 0:
                cost, count = self.cost_and_count(test_inputs, test_outputs) 
                #count = self.count(test_inputs, test_outputs, weights, biases)
                cost_points.append( (j, cost)) 
                if verbose:
                    print("Generation {0}: {1} ({2} / {3})".format(j, cost, count, test_inputs.shape[1]))

        if verbose:
            print("plotting")
            x,y = zip(*cost_points)
            fig = plt.plot(x,y)
            plt.title("Layers: {0}, Training Size: {1}, Epochs: {2}, Learning Rate: {3}".format(self.layers, len(train_inputs), epochs, eta))
            plt.ylabel("Quadratic Cost")
            plt.xlabel("Epoch")
            plt.ylim([0,max([c for (_,c) in cost_points])])
            plt.xlim([0,epochs])
            plt.show()

    def cost_and_count(self, inputs, correct_outputs, weights = None, biases = None):
        '''
        Given a set of input data and correct outputs,
        compute MSE of the difference in the actual
        and expected outputs -
        inputs - each column is a new input sample
        returns MSE cost and the number of correct classifications
        '''
        
        # Feed forward with inputs, weights and biases
        outputs = self.run(inputs, weights, biases)
        
        # Return MSE of outputs
        cost = self.cost_metric.cost(outputs, correct_outputs)
        count = self.cost_metric.count(outputs, correct_outputs)
        return cost, count
    
    def eval_layer(self, l, inputs):
        # Note adding a 1D ndarray to a 2D ndarray results in adding the i^th component
        # to the i^th column of the 2D ndarray
        #if l == 0:
        #    return inputs
        return sigmoid(np.matmul(self.weights[l], inputs) + self.biases[l][:,np.newaxis])
