# 0
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import ann


# 1
def codificar_rotulos_texto(rotulos_texto: pd.Series) -> tuple[jnp.ndarray, dict]:
    """
    Converte rótulos de texto para inteiros e retorna o mapa de classes.
    """
    classes_unicas = list(rotulos_texto.unique())  # type: ignore
    mapa_classes = {i: nome for i, nome in enumerate(classes_unicas)}
    rotulos_inteiros = rotulos_texto.map({nome: i for i, nome in mapa_classes.items()})
    return jnp.array(rotulos_inteiros.values), mapa_classes


# 2
df = pd.read_csv("datasets/iris.csv")
df.head()

# 3
X_df = df.drop(columns=["Id", "Species"])
y_texto = df["Species"]

print("Features (X):")
print(X_df.head(2))
print("\nRótulos (y):")
print(y_texto.head(2))

# 4
X = X_df.values.T
X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

y_inteiros, mapa_classes = codificar_rotulos_texto(y_texto)  # type: ignore
n_classes = len(mapa_classes)
y_one_hot = ann.codificar_one_hot(y_inteiros, n_classes)

print(f"Número de classes: {n_classes}")
print(f"Mapeamento de classes: {mapa_classes}")
print(f"\nShape de X normalizado: {X_norm.shape}")
print(f"Shape de y em one-hot: {y_one_hot.shape}")

# 5
X_train, X_test, y_train, y_test = train_test_split(
    X_norm.T, y_one_hot.T, test_size=0.2, random_state=42, stratify=y_inteiros
)

X_train, X_test = jnp.array(X_train.T), jnp.array(X_test.T)
y_train, y_test = jnp.array(y_train.T), jnp.array(y_test.T)

print("Shapes após a divisão:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# 6
print("--- Iniciando Treinamento da Rede Neural ---")
camadas = [X_train.shape[0], 10, 8, n_classes]

parametros_treinados = ann.treinar_rede(
    matriz_entrada=X_train,
    matriz_rotulos=y_train,
    dimensoes_camadas=camadas,
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="softmax",
    nome_funcao_erro="erro_categorial_cruzado",
    taxa_aprendizado=0.1,
    numero_epocas=1000,
    verbose=True,
)
print("--- Treinamento Concluído ---")

# 7
y_pred_test = ann.prever(
    X_test,
    parametros_treinados,
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="softmax",
)

print("Previsões realizadas no conjunto de teste.")
print(f"Primeiras 10 previsões: {y_pred_test[:10]}")

# 8
y_test_inteiros = jnp.argmax(y_test, axis=0)

print("--- Avaliação do Modelo no Conjunto de Teste ---")
acuracia = accuracy_score(y_test_inteiros, y_pred_test)
print(f"\nAcurácia: {acuracia:.2%}")  # Formata como porcentagem

print("\nRelatório de Classificação:")
print(
    classification_report(
        y_test_inteiros, y_pred_test, target_names=mapa_classes.values()
    )
)

# 9
cm = confusion_matrix(y_test_inteiros, y_pred_test)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=mapa_classes.values(),
    yticklabels=mapa_classes.values(),
)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.savefig("matriz_confusao_iris.png", dpi=150)
