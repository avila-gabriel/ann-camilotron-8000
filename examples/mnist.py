# 0
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import ann

SEMENTE_GLOBAL = 42


# 1
def ler_imagens_mnist(caminho_arquivo):
    """Lê imagens do formato de arquivo MNIST"""
    with open(caminho_arquivo, "rb") as f:
        _ = f.read(4)
        n_imagens = int.from_bytes(f.read(4), "big")
        n_linhas = int.from_bytes(f.read(4), "big")
        n_colunas = int.from_bytes(f.read(4), "big")
        buffer = f.read()
        dados = np.frombuffer(buffer, dtype=np.uint8)
        return dados.reshape(n_imagens, n_linhas * n_colunas)


def ler_rotulos_mnist(caminho_arquivo):
    """Lê rótulos do formato de arquivo MNIST"""
    with open(caminho_arquivo, "rb") as f:
        _ = f.read(4)  # Pula o 'magic number'
        _ = int.from_bytes(f.read(4), "big")
        buffer = f.read()
        dados = np.frombuffer(buffer, dtype=np.uint8)
        return dados


def carregar_mnist_local(caminho_pasta):
    """Carrega o dataset MNIST"""
    caminho_treino_img = os.path.join(caminho_pasta, "train-images.idx3-ubyte")
    caminho_treino_lbl = os.path.join(caminho_pasta, "train-labels.idx1-ubyte")
    caminho_teste_img = os.path.join(caminho_pasta, "t10k-images.idx3-ubyte")
    caminho_teste_lbl = os.path.join(caminho_pasta, "t10k-labels.idx1-ubyte")

    X_train = ler_imagens_mnist(caminho_treino_img)
    y_train = ler_rotulos_mnist(caminho_treino_lbl)
    X_test = ler_imagens_mnist(caminho_teste_img)
    y_test = ler_rotulos_mnist(caminho_teste_lbl)

    return (X_train, y_train), (X_test, y_test)


# 2
caminho_dados_mnist = "../datasets/mnist_data/"  # Ajuste se necessário

(X_train_flat, y_train_int), (X_test_int, y_test_int) = carregar_mnist_local(
    caminho_dados_mnist
)
print("Dados MNIST locais carregados com sucesso!")
print(
    f"Formato de X_train_flat: {X_train_flat.shape}, \
        Formato de y_train_int: {y_train_int.shape}"
)

X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_int / 255.0

X_train = jnp.array(X_train_norm.T)
X_test = jnp.array(X_test_norm.T)

n_classes = 10
y_train = ann.codificar_one_hot(jnp.array(y_train_int), n_classes)

print("\nShapes finais prontos para a rede:")
print(f"Formato de X_train: {X_train.shape}")
print(f"Formato de y_train: {y_train.shape}")

# 3
print("--- Iniciando Treinamento da Rede Neural para o MNIST ---")

camadas = [X_train.shape[0], 128, 64, n_classes]

parametros_treinados = ann.treinar_rede(
    matriz_entrada=X_train,
    matriz_rotulos=y_train,
    dimensoes_camadas=camadas,
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="softmax",
    nome_funcao_erro="erro_categorial_cruzado",
    taxa_aprendizado=0.1,
    numero_epocas=500,
    verbose=True,
    semente=SEMENTE_GLOBAL,
)
print("--- Treinamento Concluído ---")

# 4
y_pred_test = ann.prever(
    X_test,
    parametros_treinados,
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="softmax",
)

nomes_classes = [str(i) for i in range(10)]

print("--- Avaliação do Modelo no Conjunto de Teste ---")
acuracia = accuracy_score(y_test_int, y_pred_test)
print(f"\nAcurácia: {acuracia:.2%}")

print("\nRelatório de Classificação:")
print(classification_report(y_test_int, y_pred_test, target_names=nomes_classes))

# 5
cm = confusion_matrix(y_test_int, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=nomes_classes,
    yticklabels=nomes_classes,
)
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão - MNIST")
plt.savefig("matriz_confusao_mnist.png", dpi=150)
