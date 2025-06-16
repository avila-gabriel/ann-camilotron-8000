"""
Rede Neural Artificial usando JAX.

- Vetorizada em jax.numpy
- Backpropagação manual (sem jax.grad)
- Comentários estilo biblioteca, docstrings padrão NumPy
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax._src.typing import Array

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilidades
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def definir_semente(semente: int = 42) -> Array:
    """
    Gera chave de aleatoriedade JAX a partir de uma semente para reprodutibilidade.

    Parâmetros
    ----------
    semente : int, opcional
        Valor inteiro utilizado para inicialização do gerador de números aleatórios.

    Retorna
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

    Parâmetros
    ----------
    rotulos_inteiros : jnp.ndarray
        Array 1D contendo os rótulos (valores inteiros).
    numero_classes : int
        Número total de classes possíveis.

    Retorna
    -------
    jnp.ndarray
        Matriz one-hot de forma (numero_classes, n_amostras).
    """
    matriz_one_hot = jnp.zeros((numero_classes, rotulos_inteiros.size))
    matriz_one_hot = matriz_one_hot.at[
        rotulos_inteiros, jnp.arange(rotulos_inteiros.size)
    ].set(1.0)
    return matriz_one_hot


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Inicialização de parâmetros
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~ utilidades de inicialização ---------------------------------
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

    Parâmetros
    ----------
    dimensoes_camadas : Sequence[int]
        Sequência contendo o número de unidades de cada camada,
        incluindo camada de entrada e camada de saída.
    chave_aleatoria : Array
        Chave PRNG do JAX para geração de números aleatórios.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}, opcional
        Especifica a função de ativação das camadas ocultas; define
        o método de inicialização de pesos. Padrão é 'relu'.

    Retorna
    -------
    parametros_rede : dict[str, jnp.ndarray]
        Dicionário com parâmetros aprendidos da rede:
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Funções de ativação e derivadas
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def relu(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica a função de ativação ReLU elemento a elemento.

    Parâmetros
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (pré-ativação).

    Retorna
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

    Parâmetros
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do custo em relação à saída da ReLU.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da ReLU).

    Retorna
    -------
    jnp.ndarray
        Gradiente do custo em relação à entrada linear.
    """
    return gradiente_saida * (entrada_linear > 0)


def sigmoid(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica a função sigmóide elemento a elemento.

    Parâmetros
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (pré-ativação).

    Retorna
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

    Parâmetros
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do custo em relação à saída da sigmóide.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da sigmóide).

    Retorna
    -------
    jnp.ndarray
        Gradiente do custo em relação à entrada linear.
    """
    saida_sigmoid = sigmoid(entrada_linear)
    return gradiente_saida * saida_sigmoid * (1 - saida_sigmoid)


def softmax(entrada_linear: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica softmax por coluna (cada amostra).

    Parâmetros
    ----------
    entrada_linear : jnp.ndarray
        Array de ativações lineares (dim: classes, n_amostras).

    Retorna
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

    Parâmetros
    ----------
    probabilidades_preditas : jnp.ndarray
        Saídas da softmax (probabilidades preditas).
    codificacao_one_hot : jnp.ndarray
        Rótulos em codificação one-hot.

    Retorna
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

    Parâmetros
    ----------
    entrada_linear : jnp.ndarray
        Vetor ou matriz com os valores de entrada da camada (pré-ativação).

    Retorna
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

    Parâmetros
    ----------
    gradiente_saida : jnp.ndarray
        Gradiente do erro em relação à saída da função de ativação.
    entrada_linear : jnp.ndarray
        Valor da entrada linear (antes da ativação); não é utilizado
        no cálculo, mas incluído por compatibilidade.

    Retorna
    -------
    jnp.ndarray
        Gradiente do erro em relação à entrada da função de ativação.
    """
    return gradiente_saida


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Propagação para frente
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def propagacao_ante(
    matriz_entrada: jnp.ndarray,
    parametros_rede: dict[str, jnp.ndarray],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Executa a passagem para frente na rede, armazenando intermediários.

    Parâmetros
    ----------
    matriz_entrada : jnp.ndarray
        Dados de entrada (n_features, n_amostras).
    parametros_rede : dict[str, jnp.ndarray]
        Dicionário com todos os pesos/viéses da rede.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}
        Função de ativação das camadas ocultas.
    nome_ativacao_saida : {'sigmoid', 'softmax', 'linear'}
        Função de ativação da camada de saída.

    Retorna
    -------
    matriz_predicoes : jnp.ndarray
        Saída final da rede (probabilidades, logits ou valores contínuos).
    cache_propagacao : dict[str, jnp.ndarray]
        Intermediários (A0, Z1, A1, ...).
    """
    cache_propagacao: dict[str, jnp.ndarray] = {"A0": matriz_entrada}
    ativacoes = matriz_entrada
    numero_camadas = len(parametros_rede) // 2

    mapa_ativacao: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
        "relu": relu,
        "sigmoid": sigmoid,
        "softmax": softmax,
        "linear": linear,
    }

    for indice_camada in range(1, numero_camadas):
        entrada_linear = (
            parametros_rede[f"W{indice_camada}"] @ ativacoes
            + parametros_rede[f"b{indice_camada}"]
        )
        ativacoes = mapa_ativacao[nome_ativacao_oculta](entrada_linear)
        cache_propagacao[f"Z{indice_camada}"] = entrada_linear
        cache_propagacao[f"A{indice_camada}"] = ativacoes

    entrada_linear_saida = (
        parametros_rede[f"W{numero_camadas}"] @ ativacoes
        + parametros_rede[f"b{numero_camadas}"]
    )
    predicoes = mapa_ativacao[nome_ativacao_saida](entrada_linear_saida)
    cache_propagacao[f"Z{numero_camadas}"] = entrada_linear_saida
    cache_propagacao[f"A{numero_camadas}"] = predicoes
    return predicoes, cache_propagacao


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Funções de erro
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def erro_binario_cruzado(
    probabilidades_preditas: jnp.ndarray, rotulos_reais: jnp.ndarray
) -> float:
    """
    Calcula o erro binário cruzado para classificação binária.

    Parâmetros
    ----------
    probabilidades_preditas : jnp.ndarray
        Saída da rede após sigmóide (dim: 1, n_amostras).
    rotulos_reais : jnp.ndarray
        Rótulos reais (dim: 1, n_amostras).

    Retorna
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

    Parâmetros
    ----------
    probabilidades_preditas : jnp.ndarray
        Saídas da rede após softmax (classes, n_amostras).
    rotulos_one_hot : jnp.ndarray
        Rótulos reais (codificação one-hot).

    Retorna
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
    Mean Squared Error para regressão.
    """
    return float(jnp.mean((predicoes - rotulos_reais) ** 2))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Propagação para trás (backprop)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def propagacao_retro(
    parametros_rede: dict[str, jnp.ndarray],
    cache_propagacao: dict[str, jnp.ndarray],
    rotulos_reais: jnp.ndarray,
    *,
    nome_ativacao_oculta: str = "relu",
) -> dict[str, jnp.ndarray]:
    """
    Calcula o gradiente do erro em relação a cada parâmetro, unificando
    dZ = A - Y para saída (válido tanto p/ sigmoid quanto softmax).

    Parâmetros
    ----------
    parametros_rede : dict[str, jnp.ndarray]
        Pesos e viéses da rede.
    cache_propagacao : dict[str, jnp.ndarray]
        Intermediários da forward pass (Z1, A1, ..., ZL, AL).
    rotulos_reais : jnp.ndarray
        Rótulos verdadeiros (binário 1 x m ou one-hot n_classes x m).
    nome_ativacao_oculta : {'relu', 'sigmoid'}, opcional
        Ativação usada nas camadas ocultas.

    Retorna
    -------
    dict[str, jnp.ndarray]
        Gradientes dW1…dWL, db1…dbL normalizados por m.
    """
    gradientes_parametros: dict[str, jnp.ndarray] = {}
    numero_camadas = len(parametros_rede) // 2
    m = rotulos_reais.shape[1]

    A_saida = cache_propagacao[f"A{numero_camadas}"]
    dZ = A_saida - rotulos_reais
    gradientes_parametros[f"dW{numero_camadas}"] = (
        dZ @ cache_propagacao[f"A{numero_camadas - 1}"].T
    ) / m
    gradientes_parametros[f"db{numero_camadas}"] = (
        jnp.sum(dZ, axis=1, keepdims=True) / m
    )
    dA_prev = parametros_rede[f"W{numero_camadas}"].T @ dZ

    mapa_derivadas = {
        "relu": relu_derivada,
        "sigmoid": sigmoid_derivada,
        "linear": linear_derivada,
    }
    for camada in range(numero_camadas - 1, 0, -1):
        Zc = cache_propagacao[f"Z{camada}"]
        dZ = mapa_derivadas[nome_ativacao_oculta](dA_prev, Zc)
        gradientes_parametros[f"dW{camada}"] = (
            dZ @ cache_propagacao[f"A{camada - 1}"].T
        ) / m
        gradientes_parametros[f"db{camada}"] = jnp.sum(dZ, axis=1, keepdims=True) / m
        if camada > 1:
            dA_prev = parametros_rede[f"W{camada}"].T @ dZ

    return gradientes_parametros


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Atualização dos parâmetros
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def atualizar_parametros(
    parametros_rede: dict[str, jnp.ndarray],
    gradientes_parametros: dict[str, jnp.ndarray],
    taxa_aprendizado: float,
) -> dict[str, jnp.ndarray]:
    """
    Retorna um novo dicionário de pesos e vieses da rede, após atualização via gradiente
    descendente.

    Parâmetros
    ----------
    parametros_rede : dict[str, jnp.ndarray]
        Parâmetros atuais da rede.
    gradientes_parametros : dict[str, jnp.ndarray]
        Gradientes de cada parâmetro.
    taxa_aprendizado : float
        Passo do gradiente descendente.

    Retorna
    -------
    dict[str, jnp.ndarray]
        Novo dicionário de parâmetros após atualização.
    """
    novo_parametros: dict[str, Array] = {}
    for indice_camada in range(1, len(parametros_rede) // 2 + 1):
        novo_parametros[f"W{indice_camada}"] = (
            parametros_rede[f"W{indice_camada}"]
            - taxa_aprendizado * gradientes_parametros[f"dW{indice_camada}"]
        )
        novo_parametros[f"b{indice_camada}"] = (
            parametros_rede[f"b{indice_camada}"]
            - taxa_aprendizado * gradientes_parametros[f"db{indice_camada}"]
        )
    return novo_parametros


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop de treinamento
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
    tamanho_lote: int = 32,
    semente: int = 42,
    verbose: bool = True,
) -> dict[str, jnp.ndarray]:
    """
    Treina a rede neural artificial usando gradiente descendente e mini-batch.

    Parâmetros
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
    tamanho_lote : int, opcional
        Tamanho do mini-batch.
    semente : int, opcional
        Semente para reprodutibilidade.
    verbose : bool, opcional
        Se True, imprime progresso.

    Retorna
    -------
    parametros_rede : dict[str, jnp.ndarray]
        Parâmetros aprendidos ao final do treinamento.
    """
    chave_aleatoria = definir_semente(semente)
    parametros_rede, chave_aleatoria = inicializar_parametros_rede(
        dimensoes_camadas,
        chave_aleatoria=chave_aleatoria,
        nome_ativacao_oculta=nome_ativacao_oculta,
    )

    numero_amostras = matriz_entrada.shape[1]
    historico_erros: list[float] = []

    funcoes_erro: dict[str, Callable[[jnp.ndarray, jnp.ndarray], float]] = {
        "erro_binario_cruzado": erro_binario_cruzado,
        "erro_categorial_cruzado": erro_categorial_cruzado,
        "erro_mse": erro_mse,
    }
    calcular_erro = funcoes_erro[nome_funcao_erro]

    for epoca in range(1, numero_epocas + 1):
        chave_aleatoria, chave_sub = jax.random.split(chave_aleatoria)
        indices_embaralhados = jax.random.permutation(chave_sub, numero_amostras)
        for inicio in range(0, numero_amostras, tamanho_lote):
            indices_lote = indices_embaralhados[inicio : inicio + tamanho_lote]
            entradas_lote = matriz_entrada[:, indices_lote]
            rotulos_lote = matriz_rotulos[:, indices_lote]

            predicoes_lote, cache_propagacao = propagacao_ante(
                entradas_lote,
                parametros_rede,
                nome_ativacao_oculta=nome_ativacao_oculta,
                nome_ativacao_saida=nome_ativacao_saida,
            )
            erro_lote = calcular_erro(predicoes_lote, rotulos_lote)
            gradientes_parametros = propagacao_retro(
                parametros_rede,
                cache_propagacao,
                rotulos_lote,
                nome_ativacao_oculta=nome_ativacao_oculta,
            )
            parametros_rede = atualizar_parametros(
                parametros_rede, gradientes_parametros, taxa_aprendizado
            )
        historico_erros.append(erro_lote)
        if verbose and epoca % max(1, numero_epocas // 10) == 0:
            print(f"Época {epoca:>4}/{numero_epocas} – erro: {erro_lote:.6f}")
    return parametros_rede


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Predição
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def prever(
    matriz_entrada: jnp.ndarray,
    parametros_rede: dict[str, jnp.ndarray],
    *,
    nome_ativacao_oculta: str = "relu",
    nome_ativacao_saida: str = "sigmoid",
) -> jnp.ndarray:
    """
    Realiza predição dos rótulos utilizando os parâmetros aprendidos.

    Parâmetros
    ----------
    matriz_entrada : jnp.ndarray
        Dados de entrada (n_features, n_amostras).
    parametros_rede : dict[str, jnp.ndarray]
        Parâmetros aprendidos da rede.
    nome_ativacao_oculta : {'relu', 'sigmoid', 'tanh'}, opcional
        Função de ativação das camadas ocultas.
    nome_ativacao_saida : {'sigmoid', 'softmax', 'linear'}, opcional
        Função de ativação da camada de saída.

    Retorna
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
