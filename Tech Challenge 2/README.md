# Tech Challenge 2 - Machine Learning para Previsão do IBOVESPA

## Objetivo
Desenvolver um modelo preditivo para prever se o índice IBOVESPA vai fechar em alta ou baixa no dia seguinte, utilizando dados históricos.

## 1. Análise Exploratória dos Dados

### 1.1 Carregamento e Validação
- **Carregamento**: Dados históricos do IBOVESPA em formato CSV
- **Conversão**: Data convertida para datetime e definida como índice
- **Validação**: Verificação de duplicatas, inconsistências OHLC e valores nulos

### 1.2 Limpeza dos Dados
- **Lacunas temporais**: Identificação de fins de semana e feriados
- **Decisão**: Manter apenas dias de pregão reais (sem forward fill)
- **Outliers**: Mantidos por representarem eventos legítimos de mercado

### 1.3 Estatísticas Descritivas
- Análise de distribuições, correlações e coeficientes de variação
- Testes de estacionariedade (ADF e KPSS)
- Visualizações de séries temporais e histogramas

## 2. Engenharia de Features

### 2.1 Features Técnicas
- **Retornos**: Simples e logarítmicos
- **Médias móveis**: 5, 10, 20 períodos
- **Volatilidade**: Janelas de 5, 10, 20 dias
- **RSI**: Índice de Força Relativa
- **Bandas de Bollinger**: Superior, inferior e posição relativa

### 2.2 Features de Momentum
- **MACD**: Linha MACD, sinal e histograma
- **Estocástico**: %K e %D
- **Williams %R**: Oscilador de momentum

### 2.3 Features Temporais
- **Sazonalidade**: Dia da semana, mês, trimestre
- **Lags**: Valores defasados de preços e indicadores

### 2.4 Janela Deslizante
- Criação de sequências temporais para capturar padrões históricos
- Janela de 10 períodos para cada observação

## 3. Divisão dos Dados

### 3.1 Estratégia Cronológica
- **Teste**: Últimos 30 dias (mais recentes)
- **Validação**: 20% dos dados restantes (anterior ao teste)
- **Treino**: Restante dos dados (mais antigos)

### 3.2 Normalização
- StandardScaler aplicado nas features numéricas
- Preservação da ordem temporal

## 4. Modelagem

### 4.1 Modelos Implementados
- **Regressão Logística**: Modelo linear baseline
- **XGBoost**: Modelo ensemble baseado em árvores

### 4.2 Otimização de Hiperparâmetros
- **Grid Search**: Busca exaustiva com validação cruzada temporal
- **Métrica**: F1-Score para balancear precisão e recall
- **Validação**: TimeSeriesSplit com 3 folds

### 4.3 Parâmetros Otimizados
**Regressão Logística:**
- C: 0.01
- penalty: 'l2'
- solver: 'saga'

**XGBoost:**
- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 0.1
- reg_alpha: 0.001
- reg_lambda: 1.0

## 5. Avaliação dos Modelos

### 5.1 Métricas de Performance
- **Acurácia**: Proporção de predições corretas
- **Precisão**: Verdadeiros positivos / (VP + FP)
- **Recall**: Verdadeiros positivos / (VP + FN)
- **F1-Score**: Média harmônica entre precisão e recall
- **AUC-ROC**: Área sob a curva ROC

### 5.2 Resultados
- **Regressão Logística**: F1-Score CV = 0.5303
- **XGBoost**: F1-Score CV = 0.5506

### 5.3 Análise de Features
- Importância das features no modelo XGBoost
- Identificação dos indicadores mais relevantes

## 6. Validação Final

### 6.1 Teste em Dados Não Vistos
- Avaliação nos últimos 30 dias de dados
- Comparação de performance entre modelos
- Análise de erros e casos extremos

### 6.2 Interpretabilidade
- Análise das features mais importantes
- Compreensão dos padrões capturados pelos modelos

## 7. Conclusões

### 7.1 Performance dos Modelos
- XGBoost apresentou melhor performance geral
- Ambos os modelos superaram o baseline aleatório (50%)
- Performance modesta reflete a natureza caótica dos mercados financeiros

### 7.2 Limitações
- Hipótese do Mercado Eficiente limita previsibilidade
- Dados históricos podem não capturar mudanças estruturais
- Necessidade de atualização constante dos modelos

### 7.3 Próximos Passos
- Incorporação de dados externos (notícias, indicadores econômicos)
- Modelos mais complexos (LSTM, Transformer)
- Estratégias de ensemble e stacking
- Backtesting com custos de transação

## Estrutura do Projeto

```
Tech Challenge 2/
├── final/
│   └── tech_challenge_2_final.ipynb    # Notebook principal
├── dados_ibovespa_exemplo.csv          # Dataset
└── README.md                           # Este arquivo
```

## Tecnologias Utilizadas

- **Python**: Linguagem principal
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Scikit-learn**: Machine learning
- **XGBoost**: Gradient boosting
- **Matplotlib/Seaborn**: Visualizações
- **pandas_ta**: Indicadores técnicos

## Como Executar

1. Instalar dependências: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn pandas_ta`
2. Executar o notebook `tech_challenge_2_final.ipynb`
3. Os modelos serão treinados e avaliados automaticamente