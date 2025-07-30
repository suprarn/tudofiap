# Resumo do Notebook: Machine Learning para Previsão do IBOVESPA

## Objetivo
Desenvolver um modelo preditivo para prever se o índice IBOVESPA vai fechar em alta ou baixa no dia seguinte, utilizando dados históricos.

## 1. Análise Exploratória dos Dados

### 1.1 Carregamento e Validação
- **Dataset**: 3.592 registros históricos do IBOVESPA (2011-2025)
- **Colunas**: Data, Último, Abertura, Máxima, Mínima, Volume, Variação%
- **Integridade**: Sem duplicatas, sem inconsistências OHLC, apenas 1 valor nulo no volume

### 1.2 Limpeza dos Dados
- **Lacunas temporais**: 1.701 datas ausentes (1.512 fins de semana, 189 feriados)
- **Decisão**: Manter apenas dias de pregão reais (sem preenchimento)
- **Outliers**: Mantidos por representarem eventos legítimos de mercado

### 1.3 Estatísticas Descritivas
- **Preços**: Média de 83,73 pontos, desvio padrão de 28,96
- **Coeficiente de variação**: ~34,6% (alta volatilidade)
- **Volume**: Média de 0,35 bilhões, máximo de 24,87 bilhões

### 1.4 Análise dos Retornos
- **Retornos logarítmicos**: 3.591 observações
- **Média**: -0,000191 (ligeiramente negativa)
- **Desvio padrão**: 0,014896 (1,49% de volatilidade diária)
- **Assimetria**: 0,798 (distribuição assimétrica positiva)
- **Curtose**: 12,27 (caudas gordas - leptocúrtica)
- **Teste Jarque-Bera**: Rejeita normalidade (p < 0,05)

### 1.5 Volatility Clustering
- Evidência clara de agrupamento de volatilidade
- Períodos de alta volatilidade seguidos por períodos similares
- Volatilidade móvel de 30 dias mostra padrões cíclicos

## 2. Engenharia de Features

### 2.1 Features Técnicas Implementadas
- **Retornos**: Simples e logarítmicos
- **Médias móveis**: 5, 10, 20 períodos
- **Volatilidade**: Janelas de 5, 10, 20 dias
- **RSI**: Índice de Força Relativa
- **Bandas de Bollinger**: Superior, inferior e posição relativa
- **MACD**: Linha MACD, sinal e histograma
- **Estocástico**: %K e %D
- **Williams %R**: Oscilador de momentum

### 2.2 Features Temporais
- **Sazonalidade**: Dia da semana, mês, trimestre
- **Lags**: Valores defasados de preços e indicadores
- **Janela deslizante**: Sequências de 10 períodos

### 2.3 Variável Target
- **Definição**: 1 se fechamento > abertura (alta), 0 caso contrário (baixa)
- **Distribuição**: Aproximadamente balanceada
- **Prevenção de Data Leakage**: A variável target é criada usando apenas informações do mesmo dia (abertura vs fechamento), evitando o uso de informações futuras que não estariam disponíveis no momento da predição

## 3. Divisão dos Dados

### 3.1 Estratégia Cronológica
- **Teste**: Últimos 30 dias (mais recentes)
- **Validação**: 20% dos dados restantes
- **Treino**: Dados mais antigos (respeitando ordem temporal)

### 3.2 Normalização e Prevenção de Data Leakage
- **StandardScaler** aplicado nas features numéricas
- **Preservação da ordem temporal**: Divisão cronológica rigorosa para evitar data leakage
- **Validação temporal**: Uso de TimeSeriesSplit que respeita a sequência temporal
- **Features defasadas**: Uso de lags para garantir que apenas informações passadas sejam utilizadas
- **Target engineering**: Variável target construída sem usar informações futuras

## 4. Modelagem

### 4.1 Modelos Implementados
- **Regressão Logística**: Modelo linear baseline
- **XGBoost**: Modelo ensemble baseado em árvores

### 4.2 Otimização de Hiperparâmetros
- **Grid Search** com validação cruzada temporal (TimeSeriesSplit)
- **Métrica principal**: F1-Score (balanceia precisão e recall)
- **Validação**: 3 folds temporais

### 4.3 Parâmetros Otimizados

**Regressão Logística:**
- C: 0.01 (regularização forte)
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

## 5. Resultados

### 5.1 Performance dos Modelos
- **Regressão Logística**: F1-Score CV = 0.5303
- **XGBoost**: F1-Score CV = 0.5506 (melhor performance)

### 5.2 Métricas Avaliadas
- **Acurácia**: Proporção de predições corretas
- **Precisão**: Verdadeiros positivos / (VP + FP)
- **Recall**: Verdadeiros positivos / (VP + FN)
- **F1-Score**: Média harmônica entre precisão e recall
- **AUC-ROC**: Área sob a curva ROC

### 5.3 Análise de Features
- Features mais importantes identificadas no XGBoost
- Indicadores técnicos mostraram relevância significativa

## 6. Validação Final

### 6.1 Teste em Dados Não Vistos
- Avaliação nos últimos 30 dias
- Comparação entre modelos
- Análise de casos extremos

### 6.2 Interpretabilidade
- XGBoost forneceu insights sobre importância das features
- Padrões capturados pelos modelos analisados

## 7. Conclusões

### 7.1 Performance
- **XGBoost** apresentou melhor performance geral
- Ambos os modelos superaram baseline aleatório (50%)
- Performance modesta reflete natureza caótica dos mercados

### 7.2 Limitações Identificadas
- **Hipótese do Mercado Eficiente**: Limita previsibilidade
- **Dados históricos**: Podem não capturar mudanças estruturais
- **Necessidade de atualização**: Modelos requerem retreinamento constante

### 7.3 Próximos Passos Sugeridos
- Incorporar dados externos (notícias, indicadores econômicos)
- Testar modelos mais complexos (LSTM, Transformer)
- Implementar estratégias de ensemble
- Realizar backtesting com custos de transação
- Avaliar performance em diferentes regimes de mercado

## 8. Aspectos Técnicos

### 8.1 Qualidade dos Dados
- **Distribuição dos retornos**: Não-normal com caudas gordas
- **Volatility clustering**: Presente e modelado
- **Outliers**: Mantidos por representarem eventos reais

### 8.2 Metodologia Robusta e Prevenção de Data Leakage
- **Validação temporal rigorosa**: TimeSeriesSplit garante que dados futuros não sejam usados no treino
- **Divisão cronológica**: Teste com últimos 30 dias, validação com dados anteriores ao teste
- **Features lag**: Uso sistemático de defasagens para evitar look-ahead bias
- **Target sem data leakage**: Variável target criada apenas com informações do mesmo período
- **Métricas balanceadas**: F1-Score para classes desbalanceadas
- **Otimização sistemática**: Grid search com cross-validation temporal

### 8.3 Interpretabilidade
- Modelos fornecem insights sobre fatores importantes
- Features técnicas mostraram relevância
- Análise de importância das variáveis realizada

## Considerações Finais

O projeto demonstra uma abordagem metodologicamente sólida para previsão de direção do IBOVESPA, com performance modesta mas superior ao acaso. Os resultados refletem a natureza desafiadora da previsão de mercados financeiros, onde pequenas vantagens estatísticas podem ser valiosas quando aplicadas consistentemente.

A implementação seguiu boas práticas de machine learning para séries temporais, incluindo validação temporal adequada, tratamento cuidadoso dos dados e otimização sistemática de hiperparâmetros.