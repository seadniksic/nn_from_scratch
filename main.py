import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return x*(1-x)

class Neuron:
    activation = None
    bias = None

class Layer:
    neurons = np.array([])
    weights = np.array([])
    biases  = np.array([])
    activations = np.array([])
    error = np.array([])

    def __init__(self, num_nodes):
        for i in range(num_nodes):
            new_neuron = Neuron()
            self.neurons = np.append(self.neurons, new_neuron)
    
    def update_neuron_activations(self, x):
        #x is assumed to be np array
        if (x.size != self.neurons.size):
            return
        else:
            x = x.flatten()
            self.activations = x
            for i, a in enumerate(x):
                self.neurons[i].activation = a


#Dense neural net
class NN:
    layers = []
    parameters = np.array([])

    def __init__(self):
        self.layers = []
    
    def add_layer(self, num_nodes): 
        if len(self.layers) > 0:
            #When a layer is added, the previous layer's weights and biases are updated (connecting them)
            prev_num_nodes = self.layers[-1].neurons.size

            #Randomly initialize weight and bias values
            self.layers[-1].weights = np.random.uniform(low=-1, high=1, size=(num_nodes, prev_num_nodes))
            self.layers[-1].biases = np.random.uniform(low=-1, high=1, size=(num_nodes, 1))

            #Add new weights and biases to parameters list 
            self.parameters = np.concatenate((self.parameters, self.layers[-1].weights.flatten()))
            self.parameters = np.concatenate((self.parameters, self.layers[-1].biases.flatten()))

        #Create and add new layer
        new_layer = Layer(num_nodes)
        self.layers.append(new_layer)

    #def calc_grad():


    def feed_forward(self, starting_data):

        #Input data to network
        self.layers[0].update_neuron_activations(starting_data)

        for i, layer in enumerate(self.layers):

            if layer == self.layers[-1]:    #Don't propagate if final layer
                break
            
            #Get useful values for propagation
            next_layer = self.layers[i+1]
            activations = layer.activations.reshape(layer.weights.shape[1], 1)

            #Calculate new activations for next layer (and update)
            z = (np.dot(layer.weights, activations) + layer.biases).flatten()
            a = sigmoid(z)
            next_layer.update_neuron_activations(a)
        print(self.parameters.size)
        
        return self.layers[-1].activations

    def train(self, data, epochs, batch_size, learning_rate):
    #split data into batches - end up with 3d np array indexed: batches, examples, data

    #data.reshape((1, round(data.shape[0] / batch_size), batch_size))

        for epoch in epochs:
            for batch in batch_data:
                gradient = np.zeros((parameters.size))
                for example in batch:
                    output = feed_forward(data)
                    gradient += calc_grad(output)
                gradient /= batch_size
                parameters = parameters - (learning_rate * gradient)  


def main():

    # train_labels = []
    # test_labels = []
    test_data = np.loadtxt('mnist_test.csv', delimiter=',')
    # train_data = np.loadtxt('mnist_train.csv', delimiter=',')

    # test_labels = test_data[:,0:1:1].flatten()
    # train_labels = train_data[:,0:1:1].flatten()
    # test_data = test_data[:,1:]
    # train_data = train_data[:,1:]
    #Create stand in data
    batch_size = 64
    temp1 = []
    temp2 = np.array([])

    print(test_data.reshape((1, round(test_data.shape[0] / batch_size), batch_size)))
    test = np.random.rand(28,28)

    #Create Neural Network

    net = NN()

    net.add_layer(784)
    net.add_layer(16)
    net.add_layer(16)
    net.add_layer(10)
    output = net.feed_forward(test)

    print(output)
    #output = net.train() #data, epochs, batch_size, learning_rate

    #print(output)


    # for i in range
    # net.feed_forward()

    #



main()