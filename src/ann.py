"""
Rede Neural Artificial

- Vetorizada em jax.numpy
- Backpropagação manual (sem jax.grad)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax._src.typing import Array


def definir_semente(semente: int = 42) -> Array:
    """
    Gera chave de aleatoriedade JAX a partir de uma semente para reprodutibilidade.

    Parameters
    ----------
    semente : int, opcional
        Valor inteiro utilizado para inicialização do gerador de números aleatórios.

    Returns
    -------
    jax.random.PRNGKey
        Chave para geração de números aleatórios no JAX.
    """
    return jax.random.key(semente)


def codificar_one_hot(
    rotulos_inteiros: jnp.ndarray, numero_classes: int
) -> jnp.ndarray:
    """
    Converte um vetor de rótulos inteiros para codificação one-hot.

    Parameters
    ----------
    rotulos_inteiros : jnp.ndarray
        Array 1D contendo os rótulos (valores inteiros).
    numero_classes : int
        Número total de classes possíveis.

    Returns
    -------
    jnp.ndarray
        Matriz one-hot de forma (numero_classes, n_amostras).
    """
    matriz = jnp.zeros((numero_classes, rotulos_inteiros.size))
    return matriz.at[rotulos_inteiros, jnp.arange(rotulos_inteiros.size)].set(1.0)


def inicializar_parametros_rede(
    dimensoes_camadas: Sequence[int],
    *,
    chave_aleatoria: Array,
    nome_ativacao_oculta: str = "relu",
) -> tuple[dict[str, jnp.ndarray], Array]:
    """
    Inicializa pesos e vieses de todas as camadas da rede neural.

    A estratégia de inicialização varia conforme a ativação oculta:
    - Para ReLU: inicialização de He (√(2/fan_in)).
    - Para outras ativações (sigmoid, tanh): inicialização de Xavier (√(1/fan_in)).
    Além disso, o viés da primeira camada é inicializado com um valor
    pequeno e positivo (0.01) quando se usa ReLU, para evitar “neurônios
    mortos” no início.

    Parameters
    ----------
    dimensoes_camadas : Sequence[int]
        Sequência contendo o número de unidades de cada camada,
        incluindo camada de entrada e camada de saída.
    chave_aleatoria : Array
        Chave PRNG do JAX para geração de números aleatórios.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}, opcional
        Especifica a função de ativação das camadas ocultas; define
        o método de inicialização de pesos. Padrão é 'relu'.

    Returns
    -------
    parametros_rede : dict[str, jnp.ndarray]
        Dicionário com Parameters aprendidos da rede:
        - W1, b1, W2, b2, … até WL, bL.
        Cada W{i} tem forma (dimensoes_camadas[i], dimensoes_camadas[i-1]),
        e cada b{i} tem forma (dimensoes_camadas[i], 1).
    chave_aleatoria : Array
        Nova chave PRNG resultante após as divisões, para uso em
        chamadas subsequentes.
    """
    parametros: dict[str, jnp.ndarray] = {}
    for i in range(1, len(dimensoes_camadas)):
        chave_aleatoria, subkey = jax.random.split(chave_aleatoria)
        fan_in, fan_out = dimensoes_camadas[i - 1], dimensoes_camadas[i]
        if nome_ativacao_oculta == "relu":
            limite = jnp.sqrt(2.0 / fan_in)  # He
        else:
            limite = jnp.sqrt(1.0 / fan_in)  # Xavier
        parametros[f"W{i}"] = jax.random.normal(subkey, (fan_out, fan_in)) * limite
        parametros[f"b{i}"] = (
            0.01 if (i == 1 and nome_ativacao_oculta == "relu") else 0.0
        ) * jnp.ones((fan_out, 1))
    return parametros, chave_aleatoria


# activations
def relu(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica a função de ativação ReLU elemento a elemento.

    Parameters
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (pré-ativação).

    Returns
    -------
    jnp.ndarray
        Array com ReLU aplicada elemento a elemento.
    """
    return jnp.maximum(0, entrada_linear)


def relu_derivada(
    gradiente_saida: jnp.ndarray, entrada_linear: jnp.ndarray
) -> jnp.ndarray:
    """
    Calcula o gradiente da ReLU.

    Parameters
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do custo em relação à saída da ReLU.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da ReLU).

    Returns
    -------
    jnp.ndarray
        Gradiente do custo em relação à entrada linear.
    """
    return gradiente_saida * (entrada_linear > 0)


def sigmoid(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica a função sigmóide elemento a elemento.

    Parameters
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (pré-ativação).

    Returns
    -------
    jnp.ndarray
        Array com sigmóide aplicada elemento a elemento.
    """
    return 1 / (1 + jnp.exp(-entrada_linear))


def sigmoid_derivada(
    gradiente_saida: jnp.ndarray, entrada_linear: jnp.ndarray
) -> jnp.ndarray:
    """
    Calcula o gradiente da função sigmóide.

    Parameters
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do custo em relação à saída da sigmóide.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da sigmóide).

    Returns
    -------
    jnp.ndarray
        Gradiente do custo em relação à entrada linear.
    """
    saida_sigmoid = sigmoid(entrada_linear)
    return gradiente_saida * saida_sigmoid * (1 - saida_sigmoid)


def softmax(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica softmax por coluna (cada amostra).

    Parameters
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (dim: classes, n_amostras).

    Returns
    -------
    jnp.ndarray
        Distribuição de probabilidade para cada amostra (coluna).
    """
    exp_shift = jnp.exp(entrada_linear - jnp.max(entrada_linear, axis=0, keepdims=True))
    return exp_shift / jnp.sum(exp_shift, axis=0, keepdims=True)


def softmax_derivada(
    probabilidades_preditas: jnp.ndarray, codificacao_one_hot: jnp.ndarray
) -> jnp.ndarray:
    """
    Calcula o gradiente do erro categorial cruzado para softmax (saída).

    Parameters
    ----------
    probabilidades_preditas : jnp.ndarray
        Saídas da softmax (probabilidades preditas).
    codificacao_one_hot : jnp.ndarray
        Rótulos em codificação one-hot.

    Returns
    -------
    jnp.ndarray
        Gradiente da perda em relação à entrada linear da camada de saída.
    """
    return probabilidades_preditas - codificacao_one_hot


def linear(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica a função de ativação linear (identidade).

    Esta função é utilizada principalmente em problemas de regressão,
    onde não se deseja transformar a saída da rede, apenas propagá-la
    diretamente como predição contínua.

    Parameters
    ----------
    entrada_linear : jnp.ndarray
        Vetor ou matriz com os valores de entrada da camada (pré-ativação).

    Returns
    -------
    jnp.ndarray
        Mesmo array de entrada, sem alterações.
    """
    return entrada_linear


def linear_derivada(
    gradiente_saida: jnp.ndarray, entrada_linear: jnp.ndarray
) -> jnp.ndarray:
    """
    Calcula o gradiente da função de ativação linear (identidade).

    Como a função linear é definida por f(x) = x, sua derivada é 1.
    Logo, o gradiente em relação à entrada linear é simplesmente igual
    ao gradiente da camada posterior.

    Parameters
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do erro em relação à saída da função de ativação.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da ativação); não é utilizado
        no cálculo, mas incluído por compatibilidade.

    Returns
    -------
    jnp.ndarray
        Gradiente do erro em relação à entrada da função de ativação.
    """
    return gradiente_saida


def tanh(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica a função de ativação tangente hiperbólica (tanh) elemento a elemento.

    A tanh produz saídas no intervalo (-1, 1), centradas na origem, o que pode
    acelerar o aprendizado em redes profundas ao reduzir o viés de média das
    ativações.

    Parameters
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (pré-ativação).

    Returns
    -------
    jnp.ndarray
        Array com tanh aplicada elemento a elemento.
    """
    return jnp.tanh(entrada_linear)


def tanh_derivada(
    gradiente_saida: jnp.ndarray, entrada_linear: jnp.ndarray
) -> jnp.ndarray:
    """
    Calcula o gradiente da função tanh.

    A derivada da tanh é 1 - tanh²(x). Multiplicamos esse fator pelo
    gradiente vindo da camada posterior para obter o gradiente em
    relação à entrada linear.

    Parameters
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do custo em relação à saída da tanh.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da tanh).

    Returns
    -------
    jnp.ndarray
        Gradiente do custo em relação à entrada linear.
    """
    saida_tanh = tanh(entrada_linear)
    return gradiente_saida * (1 - saida_tanh**2)


_ATIVACOES: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "linear": linear,
}

_DERIVADAS: dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    "relu": relu_derivada,
    "sigmoid": sigmoid_derivada,
    "tanh": tanh_derivada,
    "linear": linear_derivada,
}


# forward-propagation
def propagacao_ante(
    matriz_entrada: jnp.ndarray,
    parametros_rede: dict[str, jnp.ndarray],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> tuple[jnp.ndarray, tuple[list[jnp.ndarray], list[jnp.ndarray]]]:
    """
    Executa a passagem para frente na rede, armazenando intermediários.

    Parameters
    ----------
    matriz_entrada : jnp.ndarray
        Dados de entrada (n_features, n_amostras).
    parametros_rede : dict[str, jnp.ndarray]
        Dicionário com todos os pesos/viéses da rede.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}
        Função de ativação das camadas ocultas.
    nome_ativacao_saida : {'sigmoid', 'softmax', 'linear'}
        Função de ativação da camada de saída.

    Returns
    -------
    matriz_predicoes : jnp.ndarray
        Saída final da rede (probabilidades, logits ou valores contínuos).
    cache_propagacao : tuple[list[jnp.ndarray], list[jnp.ndarray]]
        Intermediários (A0, Z1, A1, ...).
    """
    As: list[jnp.ndarray] = [matriz_entrada]  # A0
    Zs: list[jnp.ndarray] = []  # Z1…ZL
    num_camadas = len(parametros_rede) // 2

    # hidden layers
    for c in range(1, num_camadas):
        Z = parametros_rede[f"W{c}"] @ As[-1] + parametros_rede[f"b{c}"]
        A = _ATIVACOES[nome_ativacao_oculta](Z)
        Zs.append(Z)
        As.append(A)

    # output layer
    ZL = (
        parametros_rede[f"W{num_camadas}"] @ As[-1] + parametros_rede[f"b{num_camadas}"]
    )
    AL = _ATIVACOES[nome_ativacao_saida](ZL)
    Zs.append(ZL)
    As.append(AL)

    return AL, (As, Zs)


def erro_binario_cruzado(
    probabilidades_preditas: jnp.ndarray, rotulos_reais: jnp.ndarray
) -> float:
    """
    Calcula o erro binário cruzado para classificação binária.

    Parameters
    ----------
    probabilidades_preditas : jnp.ndarray
        Saída da rede após sigmóide (dim: 1, n_amostras).
    rotulos_reais : jnp.ndarray
        Rótulos reais (dim: 1, n_amostras).

    Returns
    -------
    float
        Valor escalar do erro médio.
    """
    eps = 1e-12
    return float(
        jnp.mean(
            -(
                rotulos_reais * jnp.log(probabilidades_preditas + eps)
                + (1 - rotulos_reais) * jnp.log(1 - probabilidades_preditas + eps)
            )
        )
    )


def erro_categorial_cruzado(
    probabilidades_preditas: jnp.ndarray, rotulos_one_hot: jnp.ndarray
) -> float:
    """
    Calcula o erro categorial cruzado para classificação multiclasse.

    Parameters
    ----------
    probabilidades_preditas : jnp.ndarray
        Saídas da rede após softmax (classes, n_amostras).
    rotulos_one_hot : jnp.ndarray
        Rótulos reais (codificação one-hot).

    Returns
    -------
    float
        Valor escalar do erro médio.
    """
    eps = 1e-12
    return float(
        -jnp.sum(rotulos_one_hot * jnp.log(probabilidades_preditas + eps))
        / rotulos_one_hot.shape[1]
    )


def erro_mse(predicoes: jnp.ndarray, rotulos_reais: jnp.ndarray) -> float:
    """
    Calcula o erro quadrático médio (Mean Squared Error) para tarefas de regressão.

    Parameters
    ----------
    predicoes : jnp.ndarray
        Valores preditos pelo modelo (dim: 1, n_amostras ou similar).
    rotulos_reais : jnp.ndarray
        Valores reais esperados (dim: 1, n_amostras ou similar).

    Returns
    -------
    float
        Valor escalar do erro quadrático médio.
    """
    return float(jnp.mean((predicoes - rotulos_reais) ** 2))


# back-propagation
def propagacao_retro(
    parametros_rede: dict[str, jnp.ndarray],
    cache_propagacao: tuple[list[jnp.ndarray], list[jnp.ndarray]],
    rotulos_reais: jnp.ndarray,
    *,
    nome_ativacao_oculta: str = "relu",
) -> dict[str, jnp.ndarray]:
    """
    Calcula o gradiente do erro em relação a cada parâmetro, unificando
    dZ = A - Y para saída (válido tanto p/ sigmoid quanto softmax).

    Parameters
    ----------
    parametros_rede : dict[str, jnp.ndarray]
        Pesos e viéses da rede.
    cache_propagacao : tuple[list[jnp.ndarray], list[jnp.ndarray]]
        Intermediários da forward pass (Z1, A1, ..., ZL, AL).
    rotulos_reais : jnp.ndarray
        Rótulos verdadeiros (binário 1 x m ou one-hot n_classes x m).
    nome_ativacao_oculta : {'relu', 'sigmoid'}, opcional
        Ativação usada nas camadas ocultas.

    Returns
    -------
    dict[str, jnp.ndarray]
        Gradientes dW1…dWL, db1…dbL normalizados por m.
    """
    As, Zs = cache_propagacao
    grads: dict[str, jnp.ndarray] = {}
    m = rotulos_reais.shape[1]
    L = len(parametros_rede) // 2

    # output layers (dZ = AL - Y, valido p/ sigmoid ou softmax)
    dZ = As[-1] - rotulos_reais
    grads[f"dW{L}"] = (dZ @ As[-2].T) / m
    grads[f"db{L}"] = jnp.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = parametros_rede[f"W{L}"].T @ dZ

    # hidden layers (reverso)
    for c in range(L - 1, 0, -1):
        dZ = _DERIVADAS[nome_ativacao_oculta](dA_prev, Zs[c - 1])
        grads[f"dW{c}"] = (dZ @ As[c - 1].T) / m  # type: ignore
        grads[f"db{c}"] = jnp.sum(dZ, axis=1, keepdims=True) / m
        if c > 1:
            dA_prev = parametros_rede[f"W{c}"].T @ dZ

    return grads


def atualizar_parametros(
    parametros_rede: dict[str, jnp.ndarray],
    gradientes_parametros: dict[str, jnp.ndarray],
    taxa_aprendizado: float,
) -> dict[str, jnp.ndarray]:
    """
    Returns um novo dicionário de pesos e vieses da rede, após atualização via gradiente
    descendente.

    Parameters
    ----------
    parametros_rede : dict[str, jnp.ndarray]
        Parameters atuais da rede.
    gradientes_parametros : dict[str, jnp.ndarray]
        Gradientes de cada parâmetro.
    taxa_aprendizado : float
        Passo do gradiente descendente.

    Returns
    -------
    dict[str, jnp.ndarray]
        Novo dicionário de Parameters após atualização.
    """
    for c in range(1, len(parametros_rede) // 2 + 1):
        parametros_rede[f"W{c}"] -= taxa_aprendizado * gradientes_parametros[f"dW{c}"]
        parametros_rede[f"b{c}"] -= taxa_aprendizado * gradientes_parametros[f"db{c}"]
    return parametros_rede


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
    Treina a rede neural artificial usando gradiente descendente e mini-batch.

    Parameters
    ----------
    matriz_entrada : jnp.ndarray
        Dados de entrada, dimensão (n_features, n_amostras).
    matriz_rotulos : jnp.ndarray
        Rótulos, dimensão (n_classes, n_amostras) ou (1, n_amostras) para binário/regressão.
    dimensoes_camadas : Sequence[int]
        Arquitetura da rede incluindo entrada e saída.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}, opcional
        Função de ativação das camadas ocultas.
    nome_ativacao_saida : {'sigmoid', 'softmax', 'linear'}, opcional
        Função de ativação da saída.
    nome_funcao_erro : {'erro_binario_cruzado', 'erro_categorial_cruzado', 'erro_mse'}, opcional
        Nome da função de erro.
    taxa_aprendizado : float, opcional
        Taxa de atualização do gradiente.
    numero_epocas : int, opcional
        Número de épocas de treinamento.
    semente : int, opcional
        Semente para reprodutibilidade.
    verbose : bool, opcional
        Se True, imprime progresso.

    Returns
    -------
    parametros_rede : dict[str, jnp.ndarray]
        Parameters aprendidos ao final do treinamento.
    """
    chave = definir_semente(semente)
    parametros_rede, _ = inicializar_parametros_rede(
        dimensoes_camadas,
        chave_aleatoria=chave,
        nome_ativacao_oculta=nome_ativacao_oculta,
    )

    erros = {
        "erro_binario_cruzado": erro_binario_cruzado,
        "erro_categorial_cruzado": erro_categorial_cruzado,
        "erro_mse": erro_mse,
    }
    loss_fn = erros[nome_funcao_erro]

    for ep in range(1, numero_epocas + 1):
        pred, cache = propagacao_ante(
            matriz_entrada,
            parametros_rede,
            nome_ativacao_oculta=nome_ativacao_oculta,
            nome_ativacao_saida=nome_ativacao_saida,
        )
        grads = propagacao_retro(
            parametros_rede,
            cache,
            matriz_rotulos,
            nome_ativacao_oculta=nome_ativacao_oculta,
        )
        parametros_rede = atualizar_parametros(parametros_rede, grads, taxa_aprendizado)

        if verbose and ep % max(1, numero_epocas // 10) == 0:
            loss = loss_fn(pred, matriz_rotulos)
            print(f"Época {ep:>4}/{numero_epocas} - erro: {loss:.6f}")

    return parametros_rede


def prever(
    matriz_entrada: jnp.ndarray,
    parametros_rede: dict[str, jnp.ndarray],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> jnp.ndarray:
    """
    Realiza predição dos rótulos utilizando os Parameters aprendidos.

    Parameters
    ----------
    matriz_entrada : jnp.ndarray
        Dados de entrada (n_features, n_amostras).
    parametros_rede : dict[str, jnp.ndarray]
        Parameters aprendidos da rede.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}, opcional
        Função de ativação das camadas ocultas.
    nome_ativacao_saida : {'sigmoid', 'softmax', 'linear'}, opcional
        Função de ativação da camada de saída.

    Returns
    -------
    jnp.ndarray
        Rótulos preditos (0/1, índice da classe, ou valores contínuos).
    """
    probabilidades_preditas, _ = propagacao_ante(
        matriz_entrada,
        parametros_rede,
        nome_ativacao_oculta=nome_ativacao_oculta,
        nome_ativacao_saida=nome_ativacao_saida,
    )
    if nome_ativacao_saida == "sigmoid":
        return (probabilidades_preditas >= 0.5).astype(int)
    elif nome_ativacao_saida == "linear":
        return probabilidades_preditas
    else:
        return jnp.argmax(probabilidades_preditas, axis=0)
