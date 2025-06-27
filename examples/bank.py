import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import src.lib.nn_functional as nn

# --- load ------------------------------------------------------------------

df = pd.read_csv("datasets/bank.csv")
df["y"] = df["deposit"].map({"yes": 1, "no": 0})
df = df.drop(columns=["deposit"])

CAT_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day",
    "poutcome",
]
NUM_COLS = ["age", "balance", "campaign"]

df_cat = pd.get_dummies(df[CAT_COLS], drop_first=True).astype(float)

X_num = df[NUM_COLS].astype(float).values
X_min = X_num.min(axis=0, keepdims=True)
X_max = X_num.max(axis=0, keepdims=True)
X_num_norm = (X_num - X_min) / (X_max - X_min + 1e-8)
df_num = pd.DataFrame(X_num_norm, columns=NUM_COLS, index=df.index)

X = pd.concat([df_num, df_cat], axis=1).astype(float).values
y = df["y"].values.reshape(1, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.T, test_size=0.15, random_state=42, stratify=y.T
)

X_train = jnp.array(X_train.T)
X_test = jnp.array(X_test.T)
y_train = jnp.array(y_train.T)
y_test = jnp.array(y_test.T)

# --- model ------------------------------------------------------------------

params = nn.treinar_rede(
    matriz_entrada=X_train,
    matriz_rotulos=y_train,
    dimensoes_camadas=(X_train.shape[0], 64, 16, 1),
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="sigmoid",
    nome_funcao_erro="erro_binario_cruzado",
    taxa_aprendizado=0.01,
    numero_epocas=1000,
    semente=42,
    verbose=True,
)

# --- evaluation -------------------------------------------------------------

y_pred_train = nn.prever(
    X_train, params, nome_ativacao_oculta="relu", nome_ativacao_saida="sigmoid"
)
y_pred_test = nn.prever(
    X_test, params, nome_ativacao_oculta="relu", nome_ativacao_saida="sigmoid"
)

acc_train = accuracy_score(
    np.array(y_train).flatten(), np.array(y_pred_train).flatten()
)
acc_test = accuracy_score(np.array(y_test).flatten(), np.array(y_pred_test).flatten())

tn, fp, fn, tp = confusion_matrix(
    np.array(y_test).flatten(), np.array(y_pred_test).flatten()
).ravel()

print(f"Acurácia treino: {acc_train:.4f}")
print(f"Acurácia teste:  {acc_test:.4f}")
print(f"FP: {fp}, FN: {fn}")

cm = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=12,
        )
plt.xticks([0, 1], ["Negativo", "Positivo"])
plt.yticks([0, 1], ["Negativo", "Positivo"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Teste (Original)")
plt.tight_layout()
plt.show()
