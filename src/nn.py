import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.hidden_size, self.input_size)
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size)
        self.b2 = np.zeros((self.output_size, 1))

    # Activation functions
    def ReLU(self, z):
        return np.maximum(0, z)
    
    def ReLU_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_derivative(self, z):
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z*(1-sigmoid_z)
    
    # Loss function (Binary Cross-Entropy)
    def bce_loss(self, y):
        # Add small epsilon to avoid log(0) issues
        eps = 1e-12
        A2_clipped = np.clip(self.A2, eps, 1 - eps)
        return -np.mean(y * np.log(A2_clipped) + (1 - y) * np.log(1 - A2_clipped))
    
    # Forward pass
    def forward(self, X):
        # Activations for hidden layer
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.ReLU(self.Z1)

        # Activations for output layer
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    # Backpropagation
    def backward(self, X, y):
        m = X.shape[1]

        # Output layer gradients
        self.dZ2 = self.A2 - y
        self.dW2 = self.dZ2 @ self.A1.T / m
        self.db2 = np.sum(self.dZ2, axis=1, keepdims=True) / m

        # Hidden layer gradients
        self.dA1 = self.W2.T @ self.dZ2
        self.dZ1 = self.dA1 * self.ReLU_derivative(self.Z1)
        self.dW1 = self.dZ1 @ X.T / m
        self.db1 = np.sum(self.dZ1, axis=1, keepdims=True) / m

    # Gradient descent update
    def update_parameters(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    # Binary predictions (0 or 1)
    def predict(self, X):
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)
    
    # Training loop
    def train(self, learning_rate, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.bce_loss(y)
            self.backward(X, y)
            self.update_parameters(learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.bce_loss(y)}")
