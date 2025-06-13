import numpy as np
from .activations import relu, relu_derivative, mse_derivative, mse_loss

class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float
    ):
        # He init
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)
        return self.A1.dot(self.W2) + self.b2

    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        dZ2 = mse_derivative(y_true, y_pred)
        dW2 = self.A1.T.dot(dZ2)
        db2 = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = dZ1.sum(axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        print_every: int = 100
    ) -> None:
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = mse_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if epoch == 1 or epoch % print_every == 0:
                print(f"Epoch {epoch:>4d}, Loss = {loss:.6f}")

    def get_feature_weights(self) -> np.ndarray:
        """
        Peso efetivo de cada input
        """
        return (self.W1.dot(self.W2)).flatten()