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

        self.layers = []
        
        # Create first hidden layer (takes inputs from input layer)
        self.layers.append(Layer(num_inputs, num_hidden_layer_neurons))
         # Create remaining hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Layer(num_hidden_layer_neurons, num_hidden_layer_neurons))
        # Create output layer
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs))
    def forward(self, inputs):
        # Pass through each layer
        for layer in self.layers:
            layer_outputs = layer.forward(inputs)
            
        return layer_outputs