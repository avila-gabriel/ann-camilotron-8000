# ann-camilotron-8000

**Uma rede neural artificial simples, vetorizada com JAX e implementada com backpropagação manual.**  
Desenvolvido como um projeto acadêmico de aprendizado de máquina.

---

## Destaques

- Totalmente vetorizada com `jax.numpy`
- Inicialização com He/Xavier
- Funções de ativação: `relu`, `sigmoid`, `softmax`, `linear`
- Três funções de erro: binária, categorial, MSE
- Treinamento com mini-batch via `treinar_rede(...)`
- Backpropagação feita *na unha* (sem `jax.grad`)
- Documentação com [mkdocstrings](reference.md)

---

## Exemplos de Uso

Explore os notebooks para ver aplicações práticas da rede em problemas como classificação binária e regressão linear.

- [Classificação bancária](examples/bank.md)
- [Regressão imobiliária](examples/real_state.md)

---

##  Sobre o projeto

Este projeto foi desenvolvido como exercício prático para compreender os fundamentos de redes neurais e backpropagação vetorizada.  
O código é curto, direto, e foi construído com foco em **clareza**, não em performance ou produção.

- Sem bibliotecas de alto nível como Keras/PyTorch
- Sem uso de `jax.grad` ou autodiff
- Ideal para fins didáticos e experimentação

---

## Código Fonte

O repositório está disponível no GitHub em:  
[github.com/avila-gabriel/ann-camilotron-8000](https://github.com/avila-gabriel/ann-camilotron-8000)

---

## Referência da API

Veja a documentação completa das funções em [API Reference](reference.md).

---

## Ambiente de Desenvolvimento

Consulte [Ambiente de Desenvolvimento](dev.md) para saber como instalar dependências, configurar o ambiente e trabalhar localmente com o projeto.

---

## Autores

- [Gabriel Avila](https://github.com/avila-gabriel)
- [Carlos Botelho](https://github.com/car2100)  
- [Juan Lopes](https://github.com/ruanmolgero)  

---

## Licença

MIT © 2025 — Projeto acadêmico para fins educacionais.
