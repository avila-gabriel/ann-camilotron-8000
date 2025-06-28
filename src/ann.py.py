from __future__ import annotations

from collections.abc import Sequence, Callable
import jax
import jax.numpy as jnp 

# ------------------------------------------------------------
#  Função para definir a semente de aleatoriedade
# ------------------------------------------------------------
def definir_semente(semente: int = 42) -> jax.random.KeyArray:
    """
    Gera uma chave (key) de aleatoriedade para JAX a partir de um número inteiro.

    Parâmetros:
    - semente (int): valor que inicializa o gerador de números aleatórios,
      garantindo que os resultados possam ser reproduzidos.

    Retorno:
    - jax.random.KeyArray: objeto interno do JAX que controla a aleatoriedade.
    """
    return jax.random.key(semente)


# ------------------------------------------------------------
#  Função para codificação one-hot de rótulos
# ------------------------------------------------------------
def codificar_one_hot(rotulos: jnp.ndarray, numero_classes: int) -> jnp.ndarray:
    """
    Converte rótulos inteiros em matriz one-hot.

    Parâmetros:
    - rotulos (jnp.ndarray): vetor 1D de inteiros (0, 1, ..., numero_classes-1).
    - numero_classes (int): quantidade total de classes distintas.

    Exemplo:
    - rotulos = [2, 0, 1], numero_classes = 3
    - retorna matriz 3x3:
      [[0,1,0],  # classe 0
       [0,0,1],  # classe 1
       [1,0,0]]  # classe 2

    Retorno:
    - jnp.ndarray de forma (numero_classes, quantidade_exemplos) com 1s na classe correta.
    """
    # Cria matriz zerada de tamanho (classes x amostras)
    matriz = jnp.zeros((numero_classes, rotulos.size))
    # Seta 1.0 na posição correspondente a cada rótulo
    return matriz.at[rotulos, jnp.arange(rotulos.size)].set(1.0)


# ------------------------------------------------------------
#  Inicialização dos parâmetros (pesos e vieses) da rede
# ------------------------------------------------------------
def inicializar_parametros_rede(
    dimensoes_camadas: Sequence[int],
    *,
    chave_aleatoria: jax.random.KeyArray,
    nome_ativacao_oculta: str = "relu",
) -> tuple[dict[str, jnp.ndarray], jax.random.KeyArray]:
    """
    Gera pesos (W) e vieses (b) para cada camada da rede.

    Parâmetros:
    - dimensoes_camadas: lista como [n_entrada, n_oculta1, ..., n_saida].
    - chave_aleatoria: objeto de aleatoriedade do JAX.
    - nome_ativacao_oculta: 'relu' ou 'sigmoid', define escala inicial.

    Processamento:
    - Para cada par de camadas, calcula limite de inicialização (He ou Xavier).
    - Gera matriz de pesos normalizada e vetor de vieses zerado.

    Retorno:
    - params: dicionário com chaves 'W1', 'b1', ..., 'Wn', 'bn'.
    - chave_aleatoria atualizada após vários splits.
    """
    params: dict[str, jnp.ndarray] = {}
    # Percorre cada conexão entre camadas
    for i in range(1, len(dimensoes_camadas)):
        chave_aleatoria, sub = jax.random.split(chave_aleatoria)
        fan_in, fan_out = dimensoes_camadas[i - 1], dimensoes_camadas[i]
        # Escolhe limite de acordo com ativação oculta (He para ReLU, Xavier para Sigmoid)
        if nome_ativacao_oculta == "relu":
            limite = jnp.sqrt(2.0 / fan_in)
        else:
            limite = jnp.sqrt(1.0 / fan_in)
        # Inicializa pesos aleatórios e vieses zerados
        params[f"W{i}"] = jax.random.normal(sub, (fan_out, fan_in)) * limite
        params[f"b{i}"] = jnp.zeros((fan_out, 1))
    return params, chave_aleatoria


# ------------------------------------------------------------
#  Funções de ativação e derivadas para backpropagation
# ------------------------------------------------------------
def relu(x: jnp.ndarray) -> jnp.ndarray:
    """Ativação ReLU: max(0, x)"""
    return jnp.maximum(0, x)

def relu_derivada(grad: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Derivada da ReLU para propagar gradientes"""
    return grad * (x > 0)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Ativação Sigmoid: valor entre 0 e 1"""
    return 1 / (1 + jnp.exp(-x))

def sigmoid_derivada(grad: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Derivada da Sigmoid multiplicada pelo gradiente de cima"""
    s = sigmoid(x)
    return grad * s * (1 - s)

def softmax(x: jnp.ndarray) -> jnp.ndarray:
    """Ativação Softmax para múltiplas classes"""
    # Subtrai maxpor coluna para evitar overflow
    e = jnp.exp(x - jnp.max(x, axis=0, keepdims=True))
    return e / jnp.sum(e, axis=0, keepdims=True)

def softmax_derivada(grad: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    S = softmax(x)  # vetor de probabilidades
    # Para cada amostra, jacobiana J = diag(S) - S @ S^T
    # Então o gradiente é grad_coluna = J @ grad
    return S * (grad - (grad * S).sum(axis=0, keepdims=True))

def linear(x: jnp.ndarray) -> jnp.ndarray:
    """Ativação Linear (sem mudança)"""
    return x

def linear_derivada(grad: jnp.ndarray, _: jnp.ndarray) -> jnp.ndarray:
    """Derivada da ativação linear é 1"""
    return grad

# Dicionários para buscar função de ativação e derivada pelo nome
_ativacoes: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu": relu,
    "sigmoid": sigmoid,
    "linear": linear,
    "softmax": softmax,
}
_derivadas: dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "relu": relu_derivada,
    "sigmoid": sigmoid_derivada,
    "linear": linear_derivada,
}

# ------------------------------------------------------------
#  Propagação para frente (forward propagation)
# ------------------------------------------------------------
def propagacao_ante(
    X: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> jnp.ndarray:
    """
    Calcula saída da rede dado input X e parâmetros.

    Parâmetros:
    - X: matriz de entrada (features) com shape (n_entrada, num_amostras).
    - params: dicionário de pesos e vieses.
    - nome_ativacao_oculta: ativação usada em camadas intermediárias.
    - nome_ativacao_saida: ativação usada na camada final.

    Retorno:
    - Saída da rede após ativação de saída.
    """
    A = X  # Ativação inicial é o próprio input
    num_camadas = len(params) // 2  # cada camada tem W e b
    # Propaga por cada camada oculta
    for c in range(1, num_camadas):
        Z = params[f"W{c}"] @ A + params[f"b{c}"]
        A = _ativacoes[nome_ativacao_oculta](Z)
    # Camada de saída
    ZL = params[f"W{num_camadas}"] @ A + params[f"b{num_camadas}"]
    return _ativacoes[nome_ativacao_saida](ZL)

# ------------------------------------------------------------
#  Funções de erro (loss)
# ------------------------------------------------------------
def erro_binario_cruzado(pred: jnp.ndarray, real: jnp.ndarray) -> float:
    """Loss Cross-Entropy binário para classificação 0/1"""
    eps = 1e-12  # evita log(0)
    return jnp.mean(-(real * jnp.log(pred + eps) + (1 - real) * jnp.log(1 - pred + eps)))

def erro_categorial_cruzado(pred: jnp.ndarray, real: jnp.ndarray) -> float:
    """Loss Cross-Entropy para múltiplas classes"""
    eps = 1e-12
    # Soma sobre classes e faz média pelas amostras
    return -jnp.sum(real * jnp.log(pred + eps)) / real.shape[1]

def erro_mse(pred: jnp.ndarray, real: jnp.ndarray) -> float:
    """Mean Squared Error para regressão"""
    return jnp.mean((pred - real) ** 2)

# ------------------------------------------------------------
#  Propagação para trás (backward propagation)
# ------------------------------------------------------------
def propagacao_retro(
    params: dict[str, jnp.ndarray],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Calcula gradientes dos parâmetros usando backpropagation.

    Parâmetros:
    - params: pesos e vieses.
    - X: entrada da rede.
    - Y: rótulos verdadeiros (one-hot ou 0/1).

    Retorno:
    - grads: dicionário com dW e db para cada camada.
    - AL: ativação final (predições antes de threshold ou argmax).
    """
    A = X
    As = [A]  # lista para armazenar ativações de cada camada
    Zs = []   # lista para armazenar valores Z de cada camada
    num_camadas = len(params) // 2
    # Feed-forward armazenando Z e A
    for c in range(1, num_camadas):
        Z = params[f"W{c}"] @ A + params[f"b{c}"]
        Zs.append(Z)
        A = _ativacoes[nome_ativacao_oculta](Z)
        As.append(A)
    # Última camada
    ZL = params[f"W{num_camadas}"] @ A + params[f"b{num_camadas}"]
    Zs.append(ZL)
    AL = _ativacoes[nome_ativacao_saida](ZL)
    As.append(AL)

    grads: dict[str, jnp.ndarray] = {}
    m = Y.shape[1]  # número de amostras
    # Gradiente da camada de saída
    dZ = AL - Y
    grads[f"dW{num_camadas}"] = (dZ @ As[-2].T) / m
    grads[f"db{num_camadas}"] = jnp.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = params[f"W{num_camadas}"].T @ dZ
    # Retropropaga pelas camadas ocultas
    for c in range(num_camadas - 1, 0, -1):
        dZ = _derivadas[nome_ativacao_oculta](dA_prev, Zs[c - 1])
        grads[f"dW{c}"] = (dZ @ As[c - 1].T) / m
        grads[f"db{c}"] = jnp.sum(dZ, axis=1, keepdims=True) / m
        if c > 1:
            dA_prev = params[f"W{c}"].T @ dZ

    return grads, AL

# ------------------------------------------------------------
#  Atualização de parâmetros (gradiente descendente)
# ------------------------------------------------------------
def atualizar_parametros(
    params: dict[str, jnp.ndarray],
    grads: dict[str, jnp.ndarray],
    taxa_aprendizado: float,
) -> dict[str, jnp.ndarray]:
    """
    Ajusta pesos e vieses subtraindo gradientes multiplicados pela taxa de aprendizado.

    Parâmetros:
    - params: valores atuais de W e b.
    - grads: gradientes dW e db calculados.
    - taxa_aprendizado: passo do gradiente (0.01, 0.001, etc.).

    Retorno:
    - novos: novo dicionário de parâmetros atualizados.
    """
    novos: dict[str, jnp.ndarray] = {}
    num_camadas = len(params) // 2
    for c in range(1, num_camadas + 1):
        novos[f"W{c}"] = params[f"W{c}"] - taxa_aprendizado * grads[f"dW{c}"]
        novos[f"b{c}"] = params[f"b{c}"] - taxa_aprendizado * grads[f"db{c}"]
    return novos

# ------------------------------------------------------------
#  Loop de treinamento (training)
# ------------------------------------------------------------
def treinar_rede(
    matriz_entrada: jnp.ndarray,
    matriz_rotulos: jnp.ndarray,
    dimensoes_camadas: Sequence[int],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
    nome_funcao_erro: str = "erro_binario_cruzado",
    taxa_aprendizado: float = 0.01,
    numero_epocas: int = 1000,
    semente: int = 42,
    verbose: bool = True,
) -> dict[str, jnp.ndarray]:
    """
    Treina a rede completa, exibindo a cada 10% de progresso o valor do erro.

    Parâmetros:
    - matriz_entrada: shape (n_entrada, m) com features normalizadas.
    - matriz_rotulos: shape (n_saida, m) com rótulos (one-hot ou 0/1).
    - dimensoes_camadas: lista de inteiros definindo tamanho das camadas.
    - nome_funcao_erro: escolha entre 'erro_binario_cruzado', 'erro_categorial_cruzado', 'erro_mse'.
    - taxa_aprendizado: passo do gradiente.
    - numero_epocas: quantas vezes percorrer todo o dataset.
    - verbose: se True mostra progresso a cada 10%.

    Retorno:
    - params finais treinados.
    """
    # Inicializa parâmetros aleatórios
    chave = definir_semente(semente)
    params, _ = inicializar_parametros_rede(
        dimensoes_camadas,
        chave_aleatoria=chave,
        nome_ativacao_oculta=nome_ativacao_oculta,
    )

    # Mapeia nomes de erro para funções
    erros: dict[str, Callable[[jnp.ndarray, jnp.ndarray], float]] = {
        "erro_binario_cruzado": erro_binario_cruzado,
        "erro_categorial_cruzado": erro_categorial_cruzado,
        "erro_mse": erro_mse,
    }
    err_fn = erros[nome_funcao_erro]

    # Loop principal de treinamento
    for ep in range(numero_epocas):
        grads, pred = propagacao_retro(
            params,
            matriz_entrada,
            matriz_rotulos,
            nome_ativacao_oculta=nome_ativacao_oculta,
            nome_ativacao_saida=nome_ativacao_saida,
        )
        # Atualiza parâmetros
        params = atualizar_parametros(params, grads, taxa_aprendizado)
        # Mostra erro periodicamente
        if verbose and (ep + 1) % max(1, numero_epocas // 10) == 0:
            loss = err_fn(pred, matriz_rotulos)
            print(f"Época {ep + 1:>4}/{numero_epocas} – erro: {loss:.6f}")
    return params

# ------------------------------------------------------------
#  Função de previsão (inference)
# ------------------------------------------------------------
def prever(
    matriz_entrada: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> jnp.ndarray:
    """
    Realiza inferência com os parâmetros treinados.

    Parâmetros:
    - matriz_entrada: dados a serem avaliados.
    - params: pesos e vieses treinados.

    Retorno:
    - Para saída sigmoid: 0 ou 1 (threshold 0.5).
    - Para saída linear: valor contínuo.
    - Para multi-classe (softmax): índice da classe mais provável.
    """
    # Calcula ativação final
    pred = propagacao_ante(
        matriz_entrada,
        params,
        nome_ativacao_oculta=nome_ativacao_oculta,
        nome_ativacao_saida=nome_ativacao_saida,
    )
    # Converte sigmoid para 0/1
    if nome_ativacao_saida == "sigmoid":
        return (pred >= 0.5).astype(int)
    # Para regressão, retorna valor bruto
    if nome_ativacao_saida == "linear":
        return pred
    # Para softmax, retorna classe com maior probabilidade
    return jnp.argmax(pred, axis=0)
