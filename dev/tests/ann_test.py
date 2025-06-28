import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ann import (
    atualizar_parametros,
    codificar_one_hot,
    definir_semente,
    erro_binario_cruzado,
    erro_categorial_cruzado,
    erro_mse,
    inicializar_parametros_rede,
    linear,
    linear_derivada,
    prever,
    propagacao_ante,
    propagacao_retro,
    relu,
    relu_derivada,
    sigmoid,
    sigmoid_derivada,
    softmax,
    treinar_rede,
)


def test_definir_semente_reprodutibilidade():
    semente = 1234
    chave_a = definir_semente(semente)
    chave_b = definir_semente(semente)
    amostras_a = jax.random.normal(chave_a, (5,))
    amostras_b = jax.random.normal(chave_b, (5,))
    np.testing.assert_array_equal(np.array(amostras_a), np.array(amostras_b))


def test_definir_semente_diferentes_sementes():
    chave_a = definir_semente(111)
    chave_b = definir_semente(222)
    amostras_a = jax.random.uniform(chave_a, (5,))
    amostras_b = jax.random.uniform(chave_b, (5,))
    assert not np.allclose(np.array(amostras_a), np.array(amostras_b)), (
        "Sequências deveriam ser diferentes para sementes distintas"
    )


def test_codificar_one_hot():
    rotulos = jnp.array([0, 2, 1, 2])
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]])
    result = codificar_one_hot(rotulos, 3)
    np.testing.assert_array_equal(np.array(result), expected)


def test_inicializar_parametros_rede_forma():
    dims = [2, 3, 1]
    key = definir_semente(42)
    params, _ = inicializar_parametros_rede(dims, chave_aleatoria=key)
    assert params["W1"].shape == (3, 2)
    assert params["b1"].shape == (3, 1)
    assert params["W2"].shape == (1, 3)
    assert params["b2"].shape == (1, 1)


def test_relu_and_derivada():
    x = jnp.array([[-1.0, 0.0, 2.0]])
    y = relu(x)
    expected = np.array([[0.0, 0.0, 2.0]])
    np.testing.assert_array_equal(np.array(y), expected)

    grad_out = jnp.ones_like(x)
    relu_grad = relu_derivada(grad_out, x)
    expected_grad = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(np.array(relu_grad), expected_grad)


def test_sigmoid_and_derivada():
    x = jnp.array([[0.0, 2.0]])
    y = sigmoid(x)
    expected = 1 / (1 + np.exp(-np.array(x)))
    np.testing.assert_allclose(np.array(y), expected, rtol=1e-6)

    grad_out = jnp.array([[1.0, 1.0]])
    sig_grad = sigmoid_derivada(grad_out, x)
    expected_grad = expected * (1 - expected)
    np.testing.assert_allclose(np.array(sig_grad), expected_grad, rtol=1e-6)


def test_softmax_cols():
    x = jnp.array([[1.0, 2.0], [2.0, 0.0]])
    s = softmax(x)
    np.testing.assert_allclose(np.sum(np.array(s), axis=0), np.ones(2), rtol=1e-6)
    assert s.shape == x.shape


def test_propagacao_ante_and_shapes():
    X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    dims = (2, 4, 1)
    key = definir_semente(0)
    params, _ = inicializar_parametros_rede(dims, chave_aleatoria=key)
    pred, (As, Zs) = propagacao_ante(X, params)
    assert pred.shape == (1, 2)
    assert As[0].shape == (2, 2)
    assert As[1].shape == (4, 2)
    assert As[2].shape == (1, 2)
    assert Zs[0].shape == (4, 2)
    assert Zs[1].shape == (1, 2)


def test_erro_binario_cruzado():
    y_true = jnp.array([[0, 1]])
    y_pred = jnp.array([[0.2, 0.7]])
    loss = erro_binario_cruzado(y_pred, y_true)
    expected = -(np.log(1 - 0.2) + np.log(0.7)) / 2
    assert np.allclose(loss, expected, rtol=1e-6)


def test_erro_categorial_cruzado():
    y_true = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    y_pred = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]).T
    loss = erro_categorial_cruzado(y_pred.T, y_true.T)
    expected = -np.mean([np.log(0.7), np.log(0.7), np.log(0.7)])
    assert np.allclose(loss, expected, rtol=1e-6)


def test_propagacao_retro_shapes():
    X = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    Y = jnp.array([[1.0, 0.0]])
    dims = (2, 4, 1)
    key = definir_semente(1)
    params, _ = inicializar_parametros_rede(dims, chave_aleatoria=key)
    pred, cache = propagacao_ante(X, params)
    grads = propagacao_retro(params, cache, Y)
    assert grads["dW1"].shape == params["W1"].shape
    assert grads["db1"].shape == params["b1"].shape
    assert grads["dW2"].shape == params["W2"].shape
    assert grads["db2"].shape == params["b2"].shape


def test_atualizar_parametros_values():
    X = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    Y = jnp.array([[1.0, 0.0]])
    dims = (2, 2, 1)
    key = definir_semente(2)
    params, _ = inicializar_parametros_rede(dims, chave_aleatoria=key)
    original = {k: np.array(params[k]) for k in params}
    pred, cache = propagacao_ante(X, params)
    grads = propagacao_retro(params, cache, Y)
    updated = atualizar_parametros(params, grads, taxa_aprendizado=0.1)
    assert updated is params
    for k in params:
        assert not np.allclose(original[k], np.array(updated[k]))


def test_treinar_rede_xor_end_to_end():
    X = jnp.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
    Y = jnp.array([[0, 1, 1, 0]])
    params = treinar_rede(
        X,
        Y,
        dimensoes_camadas=(2, 4, 1),
        numero_epocas=10,
        taxa_aprendizado=0.4,
        verbose=False,
    )
    preds = prever(X, params)
    correct = (np.array(preds) == np.array(Y)).sum()
    assert correct >= 3


def test_linear_and_derivada():
    x = jnp.array([[2.0, -3.0]])
    y = linear(x)
    np.testing.assert_array_equal(np.array(y), np.array(x))

    grad_out = jnp.array([[5.0, -1.0]])
    lin_grad = linear_derivada(grad_out, x)
    np.testing.assert_array_equal(np.array(lin_grad), np.array(grad_out))


def test_erro_mse():
    y_true = jnp.array([[1.0, 2.0]])
    y_pred = jnp.array([[1.5, 1.5]])
    loss = erro_mse(y_pred, y_true)
    expected = ((1.5 - 1.0) ** 2 + (1.5 - 2.0) ** 2) / 2
    assert np.allclose(loss, expected, rtol=1e-7)


def test_regressao_linear_end_to_end():
    X = jnp.array([[1, 2, 0, 0], [0, 0, 2, 3]], dtype=float)
    y = jnp.array([[3, 5, 5, 7]], dtype=float)
    params = treinar_rede(
        X,
        y,
        dimensoes_camadas=(2, 4, 1),
        nome_ativacao_oculta="relu",
        nome_ativacao_saida="linear",
        nome_funcao_erro="erro_mse",
        taxa_aprendizado=0.01,
        numero_epocas=2000,
        verbose=False,
    )
    y_pred = prever(
        X, params, nome_ativacao_oculta="relu", nome_ativacao_saida="linear"
    )
    mse = np.mean((np.array(y_pred) - np.array(y)) ** 2)
    assert mse < 0.1


def test_prever_multiclasse_and_regressao():
    X = jnp.eye(3)
    Y = jnp.array([[1, 0, 2]])
    Y_oh = codificar_one_hot(Y.flatten(), 3)
    params = treinar_rede(
        X,
        Y_oh,
        dimensoes_camadas=(3, 6, 3),
        nome_ativacao_oculta="relu",
        nome_ativacao_saida="softmax",
        nome_funcao_erro="erro_categorial_cruzado",
        taxa_aprendizado=0.1,
        numero_epocas=500,
        verbose=False,
    )
    y_pred = prever(
        X, params, nome_ativacao_oculta="relu", nome_ativacao_saida="softmax"
    )
    assert set(np.array(y_pred)) == {0, 1, 2}

    Xr = jnp.array([[1, 2], [2, 4]], dtype=float)
    Yr = jnp.array([[3, 7]], dtype=float)
    pr = treinar_rede(
        Xr,
        Yr,
        dimensoes_camadas=(2, 3, 1),
        nome_ativacao_oculta="relu",
        nome_ativacao_saida="linear",
        nome_funcao_erro="erro_mse",
        taxa_aprendizado=0.01,
        numero_epocas=600,
        verbose=False,
    )
    yp = prever(Xr, pr, nome_ativacao_oculta="relu", nome_ativacao_saida="linear")
    assert yp.shape == Yr.shape
    assert np.issubdtype(np.array(yp).dtype, np.floating)


def test_codificar_one_hot_missing_class():
    rotulos = jnp.array([0, 0, 1, 1])
    result = codificar_one_hot(rotulos, 4)
    assert np.all(result[2, :] == 0)
    assert np.all(result[3, :] == 0)
    assert result.shape == (4, 4)


if __name__ == "__main__":
    pytest.main(["-q", "--disable-warnings", "--maxfail=1"])
