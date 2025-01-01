import numpy as np 

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0


    def forward(self, inputs):
 
        self.output = 0
        for i, weight in zip(inputs, self.weights):
            self.output += weight*i
        
        self.output += self.bias

        return self.output

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.outputs = []
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.num_inputs))

    def forward(self, inputs):
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(inputs))

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_outputs = num_outputs

        # Now that we have all the required variables, go ahead and create the layers
        # Always remember that we do NOT need to create a layer for the inputs. The initial
        # inputs that we get make up the first input layer. So, we start from the first hidden
        # layer and create layers all the way up to the last (output) layer
        self.layers = []

        # Create the appropriate number of hidden layers each with the appropriate number of neurons
        # At the end, create the output layer

    def forward(self, inputs):
        # Take the inputs and pass those inputs to each layer in the network
        # Tip, use a for loop and one variable to keep track of the outputs of a single layer
        # Keep updating that single variable with the outputs of the layers
        # At the end, whatever is in that variable will be the output of the last layer