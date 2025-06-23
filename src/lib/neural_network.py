import numpy as np
from .activations import relu, relu_derivative, mse_derivative, mse_loss

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
        # Inicialização He dos parâmetros
        # pesos_entrada_oculta: matriz de pesos da camada de entrada → oculta
        self.pesos_entrada_oculta = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        # vieses_oculta: vetor de bias da camada oculta
        self.vieses_oculta = np.zeros((1, hidden_size))

        # pesos_oculta_saida: matriz de pesos da camada oculta → saída
        self.pesos_oculta_saida = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        # vieses_saida: vetor de bias da camada de saída
        self.vieses_saida = np.zeros((1, output_size))

        # taxa_aprendizado: learning rate (η)
        self.taxa_aprendizado = learning_rate

    def forward(self, X: np.ndarray) -> np.ndarray:
        # pre_ativacao_oculta: combinação linear entrada → oculta
        self.pre_ativacao_oculta = X.dot(self.pesos_entrada_oculta) + self.vieses_oculta
        # ativacao_oculta: aplicação da ReLU a pre_ativacao_oculta
        self.ativacao_oculta = relu(self.pre_ativacao_oculta)

        # pre_ativacao_saida: combinação linear oculta → saída (predição)
        self.pre_ativacao_saida = self.ativacao_oculta.dot(self.pesos_oculta_saida) + self.vieses_saida
        return self.pre_ativacao_saida

    def backward(self, X: np.ndarray, y_true: np.ndarray, predicoes: np.ndarray) -> None:
        # gradiente_pre_saida: ∂L/∂pre_ativacao_saida
        gradiente_pre_saida = mse_derivative(y_true, predicoes)

        # grad_pesos_oculta_saida: ∂L/∂pesos_oculta_saida
        grad_pesos_oculta_saida = self.ativacao_oculta.T.dot(gradiente_pre_saida)
        # grad_vieses_saida: ∂L/∂vieses_saida
        grad_vieses_saida = gradiente_pre_saida.sum(axis=0, keepdims=True)

        # grad_ativacao_oculta: ∂L/∂ativacao_oculta
        grad_ativacao_oculta = gradiente_pre_saida.dot(self.pesos_oculta_saida.T)
        # grad_pre_ativacao_oculta: ∂L/∂pre_ativacao_oculta
        grad_pre_ativacao_oculta = grad_ativacao_oculta * relu_derivative(self.pre_ativacao_oculta)

        # grad_pesos_entrada_oculta: ∂L/∂pesos_entrada_oculta
        grad_pesos_entrada_oculta = X.T.dot(grad_pre_ativacao_oculta)
        # grad_vieses_oculta: ∂L/∂vieses_oculta
        grad_vieses_oculta = grad_pre_ativacao_oculta.sum(axis=0, keepdims=True)

        # Atualização dos parâmetros
        self.pesos_oculta_saida    -= self.taxa_aprendizado * grad_pesos_oculta_saida
        self.vieses_saida          -= self.taxa_aprendizado * grad_vieses_saida
        self.pesos_entrada_oculta  -= self.taxa_aprendizado * grad_pesos_entrada_oculta
        self.vieses_oculta         -= self.taxa_aprendizado * grad_vieses_oculta

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, print_every: int = 100) -> None:
        for epoch in range(1, epochs + 1):
            predicoes = self.forward(X)             # saída do forward
            loss      = mse_loss(y, predicoes)     # cálculo do MSE
            self.backward(X, y, predicoes)         # backprop + atualização
            if epoch == 1 or epoch % print_every == 0:
                print(f"Epoch {epoch:>4d}, Loss = {loss:.6f}")

    def get_feature_weights(self) -> np.ndarray:
        # Calcula peso efetivo de cada input combinando as duas camadas
        # retorna vetor shape = (input_size,)
        return (self.pesos_entrada_oculta.dot(self.pesos_oculta_saida)).flatten()
