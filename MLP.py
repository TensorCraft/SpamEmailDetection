import numpy as np
from Activations import sigmoid, relu, softmax

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            input_size = layers[i][0]
            output_size = layers[i + 1][0]
            self.weights.append(np.random.randn(input_size, output_size) * 0.01)
            self.biases.append(np.zeros((1, output_size)))

    def forward(self, X):
        activations = X
        for i in range(len(self.layers) - 1):
            z = np.dot(activations, self.weights[i]) + self.biases[i]
            activations = self.layers[i + 1][1](z)
        return activations

mlp = MLP(layers = [(120, relu), (250, relu), (120, sigmoid), (10, softmax)])
X_test = np.random.randn(5, 120)

output = mlp.forward(X_test)

print("Output of the MLP forward pass:")
print(output)