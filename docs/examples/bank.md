---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Binary Classification of Term Deposit Subscription

This notebook implements a binary classification pipeline using a neural network built with JAX. The model predicts whether a customer will subscribe to a term deposit, based on features from the Bank Marketing dataset.

Dataset source: [Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import ann
```

## Load and prepare data
Target variable is `deposit`, which is mapped to 1 for 'yes' and 0 for 'no'.

```python
df = pd.read_csv("datasets/bank.csv")
df["y"] = df["deposit"].map({"yes": 1, "no": 0})
df = df.drop(columns=["deposit"])
```

## Preprocessing
- One-hot encoding for categorical columns.
- Min-max normalization for selected numeric columns.
- Concatenation of both feature types into the input matrix.

```python
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
```

## Train-test split
85% of the data is used for training and 15% for testing, with stratification based on the target label.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y.T, test_size=0.15, random_state=42, stratify=y.T
)

X_train = jnp.array(X_train.T)
X_test = jnp.array(X_test.T)
y_train = jnp.array(y_train.T)
y_test = jnp.array(y_test.T)
```

## Neural network training
Architecture: Input → 64 → 16 → 1. Output layer uses sigmoid activation for binary classification.

```python
params = ann.treinar_rede(
    matriz_entrada=X_train,
    matriz_rotulos=y_train,
    dimensoes_camadas=(X_train.shape[0], 64, 16, 1),
    nome_ativacao_oculta="relu",
    nome_ativacao_saida="sigmoid",
    nome_funcao_erro="erro_binario_cruzado",
    taxa_aprendizado=0.01,
    numero_epocas=1000,
    tamanho_lote=128,
    dropout_prob=0.0,
    semente=42,
    verbose=True,
)
```

## Performance evaluation
Includes accuracy metrics and confusion matrix for both training and test data.

```python
y_pred_train = ann.prever(
    X_train, params, nome_ativacao_oculta="relu", nome_ativacao_saida="sigmoid"
)
y_pred_test = ann.prever(
    X_test, params, nome_ativacao_oculta="relu", nome_ativacao_saida="sigmoid"
)

acc_train = accuracy_score(np.array(y_train).flatten(), np.array(y_pred_train).flatten())
acc_test = accuracy_score(np.array(y_test).flatten(), np.array(y_pred_test).flatten())

tn, fp, fn, tp = confusion_matrix(np.array(y_test).flatten(), np.array(y_pred_test).flatten()).ravel()

print(f"Training accuracy: {acc_train:.4f}")
print(f"Test accuracy:     {acc_test:.4f}")
print(f"False Positives: {fp}, False Negatives: {fn}")
```

## Confusion matrix visualization

```python
cm = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=12
        )
plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()
```
