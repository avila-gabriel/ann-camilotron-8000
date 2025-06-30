import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import ann

SEMENTE_GLOBAL = 42

# 1
print("Carregando arquivos CSV")
caminho_treino = "datasets/fashion_mnist_data/fashion-mnist_train.csv"
caminho_teste = "datasets/fashion_mnist_data/fashion-mnist_test.csv"

df_train = pd.read_csv(caminho_treino)
df_test = pd.read_csv(caminho_teste)

print("Arquivos carregados com sucesso!")
print(f"Formato do DataFrame de treino: {df_train.shape}")
print(f"Formato do DataFrame de teste:  {df_test.shape}")

# 2
y_train_int = df_train["label"].values
X_train_flat = df_train.drop(columns=["label"]).values

y_test_int = df_test["label"].values
X_test_flat = df_test.drop(columns=["label"]).values

print("Dados separados em features e rótulos.")
print(f"Shape de X_train_flat: {X_train_flat.shape}")
print(f"Shape de y_train_int: {y_train_int.shape}")

# 3
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0

X_train = jnp.array(X_train_norm.T)
X_test = jnp.array(X_test_norm.T)

n_classes = 10
y_train = ann.codificar_one_hot(jnp.array(y_train_int), n_classes)

print("Shapes finais prontos para a rede:")
print(f"Formato de X_train: {X_train.shape}")
print(f"Formato de y_train: {y_train.shape}")
print(f"Formato de X_test:  {X_test.shape}")

# 4
print("--- Iniciando Treinamento da Rede Neural ---")

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

# 5
y_pred_test = ann.prever(
    X_test,
    parametros_treinados,
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="softmax",
)

print("Previsões realizadas no conjunto de teste.")

# 6
nomes_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("--- Avaliação do Modelo no Conjunto de Teste ---")
acuracia = accuracy_score(y_test_int, y_pred_test)
print(f"\nAcurácia: {acuracia:.2%}")

print("\nRelatório de Classificação:")
print(classification_report(y_test_int, y_pred_test, target_names=nomes_classes))

# 7
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
plt.title("Matriz de Confusão - Fashion-MNIST")
plt.xticks(rotation=45)
plt.savefig("matriz_confusao_fashion_mnist.png", dpi=150)
