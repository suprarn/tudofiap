# Resumo do Notebook: Previsão do IBOVESPA com Machine Learning

## Objetivo
Desenvolver um modelo preditivo para prever se o índice IBOVESPA vai fechar em alta ou baixa no dia seguinte, com acurácia mínima de 75%.

## 1. Dados e Análise Exploratória

### Dataset
- **Fonte**: Dados históricos do IBOVESPA (1995-2025) via GitHub
- **Registros**: 7.555 observações
- **Colunas**: Data, Último, Abertura, Máxima, Mínima, Volume, Variação%
- **Qualidade**: Apenas 1 valor nulo no volume

### Análise dos Dados
- **Período**: 30 anos de dados históricos
- **Tendência**: Crescimento geral com volatilidade
- **Integridade**: Dados consistentes, sem duplicatas ou inconsistências OHLC

## 2. Engenharia de Features

### Features Criadas
- **Indicadores Técnicos**: RSI, MACD, Bandas de Bollinger, Estocástico, Williams %R
- **Médias Móveis**: 5, 10, 20 períodos
- **Volatilidade**: Janelas de 5, 10, 20 dias
- **Features Temporais**: Dia da semana, mês, trimestre
- **Lags**: Valores defasados de preços e indicadores

### Variável Target
- **Definição**: 1 se fechamento > abertura (alta), 0 caso contrário (baixa)
- **Distribuição**: Aproximadamente balanceada

## 3. Preparação dos Dados

### Divisão dos Dados
- **Teste**: Últimos 30 dias (mais recentes)
- **Validação**: 20% dos dados restantes
- **Treino**: Dados mais antigos (respeitando ordem temporal)

### Normalização
- **StandardScaler** aplicado nas features numéricas
- Preservação da ordem temporal para evitar data leakage

## 4. Modelagem

### Modelos Testados
1. **Regressão Logística**
2. **Random Forest**
3. **CatBoost**
4. **LightGBM**
5. **XGBoost**
6. **SVM**

### Otimização
- **Grid Search** com validação cruzada temporal (TimeSeriesSplit)
- **Métrica principal**: F1-Score, Acurácia, Precisão, Recall, AUC-ROC
- **Validação**: 3 folds temporais

## 5. Resultados

### Performance dos Modelos
Os modelos foram avaliados usando múltiplas métricas, com foco na acurácia para atender ao objetivo de 75%.

### Análise de Features
- Indicadores técnicos mostraram relevância significativa
- Features de momentum e volatilidade foram importantes
- Análise de importância das variáveis realizada para interpretabilidade

## 6. Funcionalidades Implementadas

### Funções de Análise
- **analisar_modelo_selecionado()**: Gera matriz de confusão e curva ROC
- **analisar_importancia_features()**: Analisa e plota importância das features

### Visualizações
- Gráfico da evolução histórica do IBOVESPA
- Matriz de confusão para avaliação dos modelos
- Curva ROC e AUC para modelos compatíveis
- Gráficos de importância das features

## 7. Metodologia

### Boas Práticas Aplicadas
- **Validação temporal**: Evita data leakage
- **Métricas balanceadas**: Múltiplas métricas para avaliação robusta
- **Otimização sistemática**: Grid search com cross-validation
- **Interpretabilidade**: Análise de importância das variáveis

### Tratamento de Dados
- Conversão adequada de tipos de dados
- Tratamento de valores nulos
- Normalização de features numéricas
- Criação de features técnicas relevantes

## 8. Considerações Técnicas

### Desafios do Mercado Financeiro
- **Natureza estocástica**: Mercados financeiros são inerentemente difíceis de prever
- **Eficiência de mercado**: Informações são rapidamente incorporadas aos preços
- **Volatilidade**: Alta variabilidade dos retornos

### Abordagem Metodológica
- Uso de indicadores técnicos consolidados
- Validação temporal rigorosa
- Múltiplos modelos para comparação
- Foco em interpretabilidade dos resultados

## Conclusão

O projeto implementa uma abordagem metodologicamente sólida para previsão de direção do IBOVESPA, utilizando técnicas modernas de machine learning e engenharia de features. A implementação inclui análise exploratória completa, múltiplos modelos de ML, otimização de hiperparâmetros e avaliação robusta com métricas apropriadas para o problema de classificação binária.

O código está estruturado de forma modular com funções para análise e visualização, facilitando a interpretação dos resultados e a tomada de decisões baseada nos modelos desenvolvidos.