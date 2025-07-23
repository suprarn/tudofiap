#  EAP - Projeto Previsão IBOVESPA Alta/Baixa

##  Resumo Executivo

**Objetivo**: Desenvolver um modelo preditivo de Machine Learning para prever se o índice IBOVESPA fechará em alta ou baixa no dia seguinte.

**Meta de Performance**: 75% de acurácia nos últimos 30 dias de dados de teste.

**Dataset**: Dados históricos do IBOVESPA de 2011 a 2025 (14+ anos) com 7 variáveis: Data, Último, Abertura, Máxima, Mínima, Volume, Variação%.

**Duração Estimada**: 8-10 semanas

**Recursos Necessários**: 1 Data Scientist, ambiente Python/R, infraestrutura de processamento.

---

##  Estrutura Analítica do Projeto (EAP)

### 1.0  PREPARAÇÃO DOS DADOS
**Duração**: 1,5 semanas 

#### 1.1 Coleta e Carregamento de Dados  
- **1.1.1** Análise do Dataset Existente  **CONCLUÍDO**
  - [ ] **1.1.1.1** Verificar integridade dos dados históricos (2011-2025)
  - [ ] **1.1.1.2** Validar formato e estrutura das colunas
  - [ ] **1.1.1.3** Documentar características do dataset
  - **Tempo**: 2 dias
  - **Entregável**: Relatório de análise inicial dos dados
  - **Critério**: Dataset carregado sem erros, estrutura documentada

- **1.1.2** Validação da Qualidade dos Dados  **CONCLUÍDO**
  - [ ] **1.1.2.1** Identificar valores ausentes (missing values)
  - [ ] **1.1.2.2** Detectar outliers e anomalias
  - [ ] **1.1.2.3** Verificar consistência temporal (datas sequenciais)
  - **Tempo**: 2 dias
  - **Entregável**: Relatório de qualidade dos dados
  - **Critério**: < 5% de dados ausentes, outliers identificados

#### 1.2 Limpeza e Pré-processamento  **CONCLUÍDO**
- **1.2.1** Definição da Métrica de Sucesso (Acuracidade > 75% no conjunto de teste)
- **1.2.2** Tratamento de Dados Ausentes
  - [ ] **1.2.2.1** Implementar estratégias de imputação para valores ausentes
  - [ ] **1.2.2.2** Validar impacto das imputações na distribuição
  - **Tempo**: 1 dia
  - **Entregável**: Dataset sem valores ausentes
  - **Critério**: 100% dos dados preenchidos adequadamente

- **1.2.3** Normalização e Padronização  **CONCLUÍDO**
  - [ ] **1.2.3.1** Converter formato de volume (remover "B", "M")
  - [ ] **1.2.3.2** Padronizar formato de percentuais
  - [ ] **1.2.3.3** Converter datas para formato datetime
  - **Tempo**: 1 dia
  - **Entregável**: Dataset padronizado
  - **Critério**: Todos os campos em formato numérico adequado

- **1.2.4** Criação da Variável Target  **CONCLUÍDO**
  - [ ] **1.2.4.1** Criar variável binária (Alta=1, Baixa=0) baseada na variação do dia seguinte
  - [ ] **1.2.4.2** Validar distribuição da variável target
  - **Tempo**: 0,5 dia
  - **Entregável**: Variável target criada
  - **Critério**: Variável binária balanceada (40-60% cada classe)

---

### 2.0  ENGENHARIA DE FEATURES
**Duração**: 2 semanas 

#### 2.1 Features Técnicas Básicas
- **2.1.1** Indicadores de Preço  **CONCLUÍDO**
  - [ ] **2.1.1.1** Calcular médias móveis (5, 10, 20, 50 dias)
  - [ ] **2.1.1.2** Criar bandas de Bollinger
  - [ ] **2.1.1.3** Calcular RSI (Relative Strength Index)
  - **Tempo**: 2 dias
  - **Entregável**: 10+ features de indicadores de preço
  - **Critério**: Features calculadas corretamente, sem valores infinitos

- **2.1.2** Indicadores de Volume  **CONCLUÍDO**
  - [ ] **2.1.2.1** Volume médio móvel (5, 10, 20 dias)
  - [ ] **2.1.2.2** Razão volume atual/volume médio
  - [ ] **2.1.2.3** Volume ponderado por preço (VWAP)
  - **Tempo**: 1 dia
  - **Entregável**: 5+ features de volume
  - **Critério**: Correlação com target > 0.1

#### 2.2 Features Avançadas
- **2.2.1** Indicadores de Volatilidade  **CONCLUÍDO**
  - [ ] **2.2.1.1** Volatilidade histórica (5, 10, 20 dias)
  - [ ] **2.2.1.2** True Range e Average True Range (ATR)
  - [ ] **2.2.1.3** Razão High-Low/Close
  - **Tempo**: 1,5 dias
  - **Entregável**: 6+ features de volatilidade
  - **Critério**: Features capturam diferentes aspectos da volatilidade

- **2.2.2** Features Temporais  **CONCLUÍDO**
  - [ ] **2.2.2.1** Dia da semana, mês, trimestre
  - [ ] **2.2.2.2** Efeitos sazonais (início/fim de mês)
  - [ ] **2.2.2.3** Lags da variável target (1, 2, 3 dias anteriores)
  - **Tempo**: 1 dia
  - **Entregável**: 8+ features temporais
  - **Critério**: Captura padrões temporais relevantes

#### 2.3 Seleção de Features  
- **2.3.1** Análise de Correlação **CONCLUÍDO**
  - [ ] **2.3.1.1** Matriz de correlação entre features
  - [ ] **2.3.1.2** Identificar features altamente correlacionadas (>0.9)
  - [ ] **2.3.1.3** Correlação com variável target
  - **Tempo**: 1 dia
  - **Entregável**: Matriz de correlação, lista de features relevantes
  - **Critério**: Features selecionadas com correlação < 0.9 entre si

- **2.3.2** Seleção Estatística **CONCLUÍDO**
  - [ ] **2.3.2.1** Aplicar testes de significância estatística
  - [ ] **2.3.2.2** Usar métodos de seleção (RFE, SelectKBest)
  - [ ] **2.3.2.3** Validar importância das features
  - **Tempo**: 1,5 dias
  - **Entregável**: Conjunto final de features (15-25 features)
  - **Critério**: Features estatisticamente significativas (p < 0.05)

---

### 3.0  DIVISÃO E VALIDAÇÃO
**Duração**: 0,5 semana

#### 3.1 Estratégia de Divisão Temporal
- **3.1.1** Divisão dos Dados **CONCLUÍDO**
  - [ ] **3.1.1.1** Treino: 2011-2023 (80% dos dados)
  - [ ] **3.1.1.2** Validação: 2024 (15% dos dados)
  - [ ] **3.1.1.3** Teste: Últimos 30 dias de 2025 (5% dos dados)
  - **Tempo**: 0,5 dia
  - **Entregável**: Datasets divididos temporalmente
  - **Critério**: Divisão respeitando ordem cronológica

#### 3.2 Validação Cruzada Temporal **CONCLUÍDO**
- **3.2.1** Configuração de Time Series Split
  - [ ] **3.2.1.1** Implementar validação cruzada temporal (5 folds)
  - [ ] **3.2.1.2** Garantir que treino sempre precede validação
  - [ ] **3.2.1.3** Configurar janela deslizante
  - **Tempo**: 1 dia
  - **Entregável**: Pipeline de validação cruzada
  - **Critério**: Validação sem vazamento de dados futuros

---

### 4.0  MODELAGEM
**Duração**: 2,5 semanas 

#### 4.1 Modelos Baseline
- **4.1.1** Modelos Simples
  - [ ] **4.1.1.1** Regressão Logística
  - [ ] **4.1.1.2** Naive Bayes
  - [ ] **4.1.1.3** K-Nearest Neighbors (KNN)
  - **Tempo**: 2 dias
  - **Entregável**: 3 modelos baseline treinados
  - **Critério**: Acurácia > 55% (melhor que random)

#### 4.2 Modelos Avançados
- **4.2.1** Ensemble Methods **CONCLUÍDO**
  - [x] **4.2.1.1** Random Forest **CONCLUÍDO**
  - [x] **4.2.1.2** Gradient Boosting (XGBoost) **CONCLUÍDO**
  - [x] **4.2.1.3** LightGBM **CONCLUÍDO**
  - **Tempo**: 3 dias
  - **Entregável**: 3 modelos ensemble treinados
  - **Critério**: Acurácia > 65%

- **4.2.2** Modelos de Deep Learning **CONCLUÍDO**
  - [x] **4.2.2.1** Rede Neural Feedforward **CONCLUÍDO**
  - [x] **4.2.2.2** LSTM para séries temporais **CONCLUÍDO**
  - [x] **4.2.2.3** CNN-LSTM híbrido **CONCLUÍDO**
  - **Tempo**: 4 dias
  - **Entregável**: 3 modelos de deep learning
  - **Critério**: Acurácia > 70%

- **Observação** Documentar a justificativa da escolha do modelo.

#### 4.3 Avaliação Inicial
- **4.3.1** Métricas de Performance
  - [ ] **4.3.1.1** Calcular acurácia, precisão, recall, F1-score
  - [ ] **4.3.1.2** Matriz de confusão para cada modelo
  - [ ] **4.3.1.3** Curva ROC e AUC
  - **Tempo**: 1 dia
  - **Entregável**: Relatório comparativo de modelos
  - **Critério**: Identificar top 3 modelos com melhor performance

---

### 5.0  OTIMIZAÇÃO
**Duração**: 1,5 semanas 

#### 5.1 Hyperparameter Tuning
- **5.1.1** Grid Search e Random Search
  - [ ] **5.1.1.1** Definir espaço de hiperparâmetros para top 3 modelos
  - [ ] **5.1.1.2** Executar Grid Search com validação cruzada
  - [ ] **5.1.1.3** Aplicar Random Search para espaços grandes
  - **Tempo**: 3 dias
  - **Entregável**: Hiperparâmetros otimizados
  - **Critério**: Melhoria de pelo menos 2% na acurácia

#### 5.2 Otimização Bayesiana
- **5.2.1** Bayesian Optimization
  - [ ] **5.2.1.1** Implementar otimização bayesiana (Optuna/Hyperopt)
  - [ ] **5.2.1.2** Otimizar hiperparâmetros do melhor modelo
  - [ ] **5.2.1.3** Validar estabilidade dos resultados
  - **Tempo**: 2 dias
  - **Entregável**: Modelo final otimizado
  - **Critério**: Acurácia consistente > 72%

#### 5.3 Ensemble Final
- **5.3.1** Combinação de Modelos
  - [ ] **5.3.1.1** Criar ensemble dos 3 melhores modelos
  - [ ] **5.3.1.2** Testar diferentes estratégias (voting, stacking)
  - [ ] **5.3.1.3** Otimizar pesos do ensemble
  - **Tempo**: 2 dias
  - **Entregável**: Modelo ensemble final
  - **Critério**: Acurácia > 74%

---

### 6.0  VALIDAÇÃO E TESTE
**Duração**: 1 semana 

#### 6.1 Validação no Conjunto de Teste
- **6.1.1** Teste Final
  - [ ] **6.1.1.1** Aplicar modelo nos últimos 30 dias (dados de teste)
  - [ ] **6.1.1.2** Calcular todas as métricas de performance
  - [ ] **6.1.1.3** Verificar se meta de 75% foi atingida
  - **Tempo**: 1 dia
  - **Entregável**: Resultados finais no conjunto de teste
  - **Critério**: Acurácia ≥ 75% nos últimos 30 dias

#### 6.2 Análise de Robustez
- **6.2.1** Testes de Estabilidade
  - [ ] **6.2.1.1** Testar modelo em diferentes períodos
  - [ ] **6.2.1.2** Análise de sensibilidade a outliers
  - [ ] **6.2.1.3** Validar performance em diferentes condições de mercado
  - **Tempo**: 2 dias
  - **Entregável**: Relatório de robustez
  - **Critério**: Performance estável (±3%) em diferentes períodos

#### 6.3 Interpretabilidade
- **6.3.1** Explicabilidade do Modelo
  - [ ] **6.3.1.1** Análise de importância das features (SHAP)
  - [ ] **6.3.1.2** Interpretação dos principais drivers
  - [ ] **6.3.1.3** Casos de uso e limitações
  - **Tempo**: 2 dias
  - **Entregável**: Relatório de interpretabilidade
  - **Critério**: Top 10 features explicadas claramente

---

### 7.0  ANÁLISE E RELATÓRIO
**Duração**: 1 semana 

#### 7.1 Análise de Resultados
- **7.1.1** Performance Detalhada
  - [ ] **7.1.1.1** Análise por período (mensal, trimestral)
  - [ ] **7.1.1.2** Performance por condições de mercado
  - [ ] **7.1.1.3** Análise de erros e falsos positivos/negativos
  - **Tempo**: 2 dias
  - **Entregável**: Análise detalhada de performance
  - **Critério**: Insights acionáveis identificados

#### 7.2 Documentação Final
- **7.2.1** Relatório Executivo
  - [ ] **7.2.1.1** Resumo executivo com principais resultados
  - [ ] **7.2.1.2** Metodologia e abordagem técnica
  - [ ] **7.2.1.3** Recomendações e próximos passos
  - **Tempo**: 2 dias
  - **Entregável**: Relatório final completo
  - **Critério**: Documento profissional de 15-20 páginas

#### 7.3 Entrega e Deploy
- **7.3.1** Preparação para Produção
  - [ ] **7.3.1.1** Código limpo e documentado
  - [ ] **7.3.1.2** Pipeline de predição automatizado
  - [ ] **7.3.1.3** Instruções de uso e manutenção
  - **Tempo**: 1 dia
  - **Entregável**: Código e pipeline prontos para produção
  - **Critério**: Sistema funcional e documentado

---

##  Cronograma Consolidado

**Duração Total**: 10 semanas

---

##  Critérios de Sucesso

### Critérios Primários
- ✅ **Acurácia ≥ 75%** nos últimos 30 dias de dados
- ✅ **Precisão e Recall balanceados** (> 70% cada)
- ✅ **AUC-ROC ≥ 0.80**

### Critérios Secundários
- ✅ **Performance estável** em diferentes períodos
- ✅ **Modelo interpretável** com features explicáveis
- ✅ **Pipeline automatizado** para predições futuras
- ✅ **Documentação completa** e código limpo

### Critérios de Qualidade
- ✅ **Código versionado** e reproduzível
- ✅ **Testes unitários** para funções críticas
- ✅ **Validação sem data leakage**
- ✅ **Relatório técnico profissional**

---

## 📈 Próximos Passos Pós-Projeto

1. **Monitoramento Contínuo**: Implementar sistema de monitoramento da performance
2. **Retreinamento**: Agendar retreinamento mensal com novos dados
3. **Expansão**: Considerar outros índices (IFIX, SMLL, etc.)
4. **Integração**: API para consumo das predições
5. **Melhoria Contínua**: Incorporar novos indicadores e features