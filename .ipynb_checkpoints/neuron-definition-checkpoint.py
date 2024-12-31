import numpy as np 

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0


    def forward(self, inputs):
        # Do the dot product calculations here

        self.output = 0
        # Continue with the dot product calculations here

        return self.output