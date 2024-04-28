# Guia de Uso

Este é um projeto de exemplo para treinar e avaliar um modelo de rede neural linear para classificação de flores usando PyTorch.

## Arquivos

1. **model.py**: Define a arquitetura do modelo.
2. **data.py**: Carrega e prepara os dados para treinamento e teste.
3. **train.py**: Treina o modelo usando os dados de treinamento.
4. **evaluate.py**: Avalia o desempenho do modelo usando os dados de teste.
5. **predict.py**: Faz previsões usando o modelo treinado.

## Como Usar

Siga estas etapas para usar o projeto:

1. **Instalação de Dependências**: Certifique-se de ter as dependências instaladas. Você pode instalá-las executando: `pip install -r requirements.txt`

2. **Execução dos Arquivos**:

   - Execute os arquivos na seguinte ordem:

     ```
     python train.py
     python predict.py
     ```

    Certifique-se de executar cada arquivo após o término do anterior para garantir que todas as dependências estejam corretamente estabelecidas.

3. **Visualização dos Resultados**: Ao executar `train.py`, um gráfico da curva de perda será exibido para visualizar o treinamento do modelo e será criado 2 arquivos: `evaluation_predictions.log` que contém os acertos e `evaluation_loss.log` ratio de loss.

4. **Previsões Personalizadas**: Ao executar `predict.py`, você pode fornecer dados personalizados para o modelo fazer previsões.

## Notas Adicionais

- Certifique-se de ter PyTorch instalado. Você pode instalar PyTorch via pip ou conda, conforme documentado em [pytorch.org](https://pytorch.org/get-started/locally/).
- Certifique-se de ter os dados necessários (por exemplo, `iris.csv`) na pasta `nn_data/`.
- Este projeto é apenas um exemplo simples desenvolvido para relembrar o basico do Pytorch usando Neural Network Linear.
