import numpy as np 
import math
import random
import time
import pickle

def activation(x):
    '''Insert desired neuron activation function here'''
    return (1/(1+np.power(math.exp(1), -x)))

def d_activation(x):
    '''Insert derivative of desired neuron activation function here'''
    return np.multiply(activation(x), (1-activation(x)))

def softmax(z, c):
    '''Input a vector of weighted inputs z and returns a vector of probabilities that sum to c'''
    x = np.power(math.exp(1), c*z) 
    return x/sum(x)

class Neural_Network(object):
    '''With softmax; L2 regularization; optimized weight initialization'''
    def __init__(self, sizes=None):
        '''Initialise the neural network. Sizes = number of layers'''
        #initialize the input layer
        self.layers = [Layer()]
        for i, size in enumerate(sizes[1:]): #generate random weights and biases for hidden layers
            weights = np.random.randn(size,sizes[i])/np.sqrt(sizes[i])
            biases = np.random.randn(size, 1)
            self.layers = self.layers + [Layer(weights, biases)]

    def train(self, training_data, epochs, mini_batch_size, learning_rate, regularization, test_data = None, save_name = False):
        '''Train the network with given data'''
        if test_data: #if specified, we will track our accuracy growth over time
            test_data_accuracies = []

        training_data = list(training_data)
        n = len(training_data)
        
        for j in range(1, epochs + 1): #run this for the number of epochs specified
            start = time.time()
            random.shuffle(training_data)
            #divide data into mini batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: #update network within each mini batch
                self.update_mini_batch(mini_batch, learning_rate, regularization, len(training_data))

            if test_data: #if test_data is true, give regular training updates / track accuracy
                test_data = list(test_data)
                total_correct = self.evaluate(test_data)
                accuracy = 100*total_correct/len(test_data)
                print("Epoch {} : {} / {} = {}% Accuracy".format(j,total_correct,len(test_data),accuracy))
                test_data_accuracies.append(accuracy)
            else:
                print("Epoch {} complete".format(j))

            end = time.time()
            print("Time Elapsed: {}".format(end - start))

        if save_name: #save network
            with open(save_name + '.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        if test_data:
            return test_data_accuracies

    def update_mini_batch(self, mini_batch, learning_rate, regularization, n):
        '''Updates the network parameters based on the samples within the mini batch'''
        #we will store the total nabla(biases) and nabla(weights) for each layer over all the samples in each mini batch
        for layer in self.layers[1:]:
            layer.nabla_b = np.zeros(layer.biases.shape)
            layer.nabla_w = np.zeros(layer.weights.shape)

        #for each sample within the mini batch
        for sample in mini_batch:
            x, y = sample[:]
            self.feed_forward(x)
            self.back_propagate(y)
        
        #update each layer's parameters with the average nabla over all samples
        for layer in self.layers[1:]:
            layer.biases = layer.biases - learning_rate * layer.nabla_b / len(mini_batch)
            layer.weights = ((1 - (learning_rate * regularization / n)) * layer.weights) - (learning_rate * layer.nabla_w / len(mini_batch))

    def feed_forward(self, x):
        '''Takes input and generates activations for each layer in the network'''
        #set the input layer's activations to equal the input
        self.layers[0].activations = x
        #for each other layer, calculate its activations based on the previous layer's activations and the layer parameters
        for layer in self.layers[1:len(self.layers)-1]:
            layer.z = layer.weights.dot(x) + layer.biases
            layer.activations = activation(layer.z) 
            x = layer.activations
        #softmax layer
        layer = self.layers[-1]
        layer.z = layer.weights.dot(x) + layer.biases
        layer.activations = softmax(layer.z, 1) 
        return layer.activations

    def back_propagate(self, y):
        '''For a given desired result, y, calculate the desired change in biases and weights'''

        #do last layer first
        L = self.layers[-1]
        error = L.activations-y #log likelihood cost for softmax layer
        L.nabla_b = L.nabla_b + error
        L.nabla_w = L.nabla_w + (error.dot(self.layers[-2].activations.T))

        #go backwards from last layer until reaching the second layer
        for i in range(len(self.layers) - 2, 0, -1):
            Lahead = self.layers[i+1]
            L = self.layers[i]
            Lbehind = self.layers[i-1]
            
            error = (np.multiply((Lahead.weights.T).dot(error), d_activation(L.z)))
            L.nabla_b = L.nabla_b + error
            L.nabla_w = L.nabla_w + (np.dot(error, Lbehind.activations.transpose()))

    def evaluate(self, test_data):
        '''Evaluates the accuracy of a network for given test data'''
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        total_correct = sum(int(x == y) for (x, y) in test_results)
        return total_correct

class Layer(object):
    def __init__(self, weights=None, biases=None):
        '''Layer object represents each layer within the network'''
        self.weights = weights
        self.biases = biases
        self.nabla_b = None
        self.nabla_w = None

