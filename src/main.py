# main.py
from lib.preprocessing   import load_and_preprocess
from lib.neural_network   import NeuralNetwork
from lib.reporting       import report_feature_weights
from lib.evaluation      import evaluate_model
from lib.plotting        import plot_results
# === 5) Main apenas chama funções ===
def main():
    # Parâmetros
    filepath = '../datasets/real_estate_dataset.csv'
    factor = 1.5
    hidden_size = 256
    learning_rate = 0.015
    epochs = 3000
    print_every = 200

    # Preprocessamento
    X_train, X_test, y_train, y_test, features, y_min, y_max = \
        load_and_preprocess(filepath, factor=factor)

    # Treinamento
    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        learning_rate=learning_rate
    )
    print("Treinamento iniciado...")
    nn.train(X_train, y_train, epochs=epochs, print_every=print_every)

    # Relatório de pesos
    weights = nn.get_feature_weights()
    report_feature_weights(weights, features)

    # Avaliação
    mse, r2, y_actual, y_pred = evaluate_model(nn, X_test, y_test, y_min, y_max)
    print(f"MSE no teste (real): {mse:.2f}")
    print(f"R² no teste: {r2:.4f}\n")

    # Plot
    plot_results(y_actual, y_pred)

if __name__ == "__main__":
    main()
