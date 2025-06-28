import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import ann

df = pd.read_csv("datasets/real_estate_dataset.csv")

X_df = df.drop(columns=["ID", "Price"])
y = df["Price"].values.reshape(1, -1)
X = X_df.values.T

X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)
y_min = y.min()
y_max = y.max()
y_norm = (y - y_min) / (y_max - y_min + 1e-8)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm.T, y_norm.T, test_size=0.2, random_state=1
)
X_train = jnp.array(X_train.T)
X_test = jnp.array(X_test.T)
y_train = jnp.array(y_train.T)
y_test = jnp.array(y_test.T)

print("Shapes de treino:", X_train.shape, y_train.shape)

params = ann.treinar_rede(
    matriz_entrada=X_train,
    matriz_rotulos=y_train,
    dimensoes_camadas=(
        X_train.shape[0],
        32,
        1,
    ),
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="linear",
    nome_funcao_erro="erro_mse",
    taxa_aprendizado=0.05,
    numero_epocas=2000,
    verbose=True,
)
y_pred_train = ann.prever(
    X_train, params, nome_ativacao_oculta="relu", nome_ativacao_saida="linear"
)
y_pred_test = ann.prever(
    X_test, params, nome_ativacao_oculta="relu", nome_ativacao_saida="linear"
)

y_pred_train_real = np.array(y_pred_train) * (y_max - y_min) + y_min
y_train_real = np.array(y_train) * (y_max - y_min) + y_min
y_pred_test_real = np.array(y_pred_test) * (y_max - y_min) + y_min
y_test_real = np.array(y_test) * (y_max - y_min) + y_min

mse_train = np.mean((y_pred_train_real - y_train_real) ** 2)
mse_test = np.mean((y_pred_test_real - y_test_real) ** 2)

# raiz quadrada para voltar à unidade “real”
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

r2_train = r2_score(y_train_real.flatten(), y_pred_train_real.flatten())
r2_test = r2_score(y_test_real.flatten(), y_pred_test_real.flatten())

print(f"MSE real treino: {rmse_train:.2f}  # (unidade $)")
print(f"MSE real teste:  {rmse_test:.2f}  # (unidade $)")
print(f"R² treino:         {r2_train:.4f}")
print(f"R² teste:          {r2_test:.4f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test_real.flatten(), y_pred_test_real.flatten(), alpha=0.7, edgecolor="k")
plt.plot(
    [y_test_real.min(), y_test_real.max()],
    [y_test_real.min(), y_test_real.max()],
    "r--",
    lw=2,
)
plt.xlabel("Preço real")
plt.ylabel("Preço predito pela RNA")
plt.title("Regressão Imobiliária: Real vs. Predito (teste)")
plt.grid(True)
plt.tight_layout()
plt.savefig("real_vs_predito.png", dpi=150)
