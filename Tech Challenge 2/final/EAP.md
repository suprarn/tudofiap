# **Estrutura Analítica do Projeto (EAP) para Previsão de Tendência do IBOVESPA**

Projeto: Desenvolvimento de Modelo de Machine Learning para Previsão da Direção Diária do IBOVESPA.

Referência: Steering.txt

### **1.0 \- Fase 1: Integridade dos Dados e Análise Exploratória (Dia 1\)**

*Este pacote de trabalho foca em garantir a qualidade dos dados brutos e extrair insights iniciais sobre o comportamento do mercado.*

* **1.1 Aquisição e Validação Inicial**  
  - [ ] 1.1.1 Carregar o dataset de 15 anos em um ambiente de análise (ex: Pandas DataFrame).  
  - [ ] 1.1.2 Converter a coluna de data para o formato datetime e defini-la como índice.  
  - [ ] 1.1.3 Realizar verificação de integridade: checar duplicatas e consistência dos dados (Máxima \>= Fechamento \>= Mínima).  
* **1.2 Limpeza e Tratamento de Dados**  
  - [ ] 1.2.1 Identificar lacunas na série temporal (dias sem pregão).  
  - [ ] 1.2.2 Aplicar a estratégia de preenchimento para a frente (forward-fill / ffill) para preencher os valores em feriados e fins de semana.  
  - [ ] 1.2.3 Documentar a decisão de não remover outliers, justificando-a pela natureza dos eventos de mercado.  
* **1.3 Análise Exploratória de Dados (EDA)**  
  - [ ] 1.3.1 Gerar e analisar estatísticas descritivas (.describe()).  
  - [ ] 1.3.2 Plotar o gráfico de linha do preço de fechamento (Close) e o gráfico de barras do volume (Volume) ao longo do tempo.  
  - [ ] 1.3.3 Calcular os retornos logarítmicos diários.  
  - [ ] 1.3.4 Plotar o histograma dos retornos para analisar a distribuição (verificar curtose e "caudas gordas").  
  - [ ] 1.3.5 Plotar o gráfico de linha dos retornos para identificar visualmente o agrupamento de volatilidade (volatility clustering).  
* **1.4 Teste de Estacionariedade**  
  - [ ] 1.4.1 Aplicar os testes ADF (Augmented Dickey-Fuller) e KPSS (Kwiatkowski-Phillips-Schmidt-Shin) na série de preços de fechamento.  
  - [ ] 1.4.2 Aplicar os testes ADF e KPSS na série de retornos diários.  
  - [ ] 1.4.3 Documentar os resultados, confirmando a não estacionariedade dos preços e a estacionariedade dos retornos.

### **2.0 \- Fase 2: Definição do Alvo e Engenharia de Atributos (Dia 2\)**

*Este pacote de trabalho foca na criação da variável a ser prevista e no enriquecimento dos dados com atributos preditivos, com atenção rigorosa para evitar viés de lookahead.*

* **2.1 Criação da Variável Alvo (Target)**  
  - [ ] 2.1.1 Implementar a lógica para criar a coluna Target: 1 se Close(t+1) \> Close(t), e 0 caso contrário.  
  - [ ] 2.1.2 Aplicar o deslocamento (.shift(-1)) na coluna Target para alinhar corretamente os atributos do dia t com o resultado do dia t+1.  
  - [ ] 2.1.3 Remover a última linha do DataFrame, que conterá um valor nulo (NaN) no Target após o deslocamento.  
* **2.2 Análise da Distribuição de Classes**  
  - [ ] 2.2.1 Calcular e analisar a frequência das classes 0 e 1 na coluna Target.  
  - [ ] 2.2.2 Documentar o nível de desbalanceamento e suas implicações para a seleção de métricas e treinamento do modelo.  
* **2.3 Engenharia de Atributos (Feature Engineering)**  
  - [ ] 2.3.1 **Atributos de Momento:** Criar colunas de retornos defasados (lags) para os últimos 5 dias (Return\_Lag\_1 a Return\_Lag\_5).  
  - [ ] 2.3.2 **Atributos de Tendência:**  
        * Calcular as Médias Móveis Simples (SMA) de 10, 20 e 50 dias.  
        * Criar atributos normalizados: Ratio\_Close\_SMA20 e Ratio\_SMA10\_SMA50.  
  - [ ] 2.3.3 **Atributos de Momento (Oscilador):** Calcular o Índice de Força Relativa (RSI) de 14 dias (RSI\_14).  
  - [ ] 2.3.4 **Atributos de Volatilidade:**  
        * Calcular as Bandas de Bollinger de 20 dias.  
        * Criar atributos derivados: BB\_Width\_20 e BB\_Position\_20.  
  - [ ] 2.3.5 **Atributos de Volume:** Calcular o On-Balance Volume (OBV) e sua variação percentual (OBV\_pct\_change).  
  - [ ] 2.3.6 Remover todas as linhas no início do DataFrame que contenham NaN resultantes do cálculo dos indicadores.  
* **2.4 Documentação dos Atributos**  
  - [ ] 2.4.1 Criar e preencher a Tabela "Dicionário de Engenharia de Atributos" conforme o guia.

### **3.0 \- Fase 3: Preparação da Base para Modelagem (Dia 3\)**

*Este pacote de trabalho estrutura os dados para que possam ser consumidos pelos algoritmos de machine learning, respeitando a ordem temporal.*

* **3.1 Estruturação com Janela Deslizante**  
  - [ ] 3.1.1 Definir a janela de entrada (lookback window) como n=5 dias.  
  - [ ] 3.1.2 Implementar a lógica para transformar a série temporal em um dataset tabular, onde cada linha contém os atributos dos últimos 5 dias e o alvo correspondente.  
* **3.2 Divisão Cronológica dos Dados**  
  - [ ] 3.2.1 Definir uma data de corte para a divisão treino-teste (ex: 80% para treino, 20% para teste).  
  - [ ] 3.2.2 Separar os dados em X\_train, y\_train, X\_test, e y\_test sem usar amostragem aleatória.  
* **3.3 Escalonamento de Atributos (Feature Scaling)**  
  - [ ] 3.3.1 Instanciar um StandardScaler.  
  - [ ] 3.3.2 Ajustar (fit) o scaler **exclusivamente** no conjunto de treino (X\_train).  
  - [ ] 3.3.3 Aplicar a transformação (transform) nos conjuntos de treino e teste.

### **4.0 \- Fase 4: Modelagem e Validação (Dias 3-4)**

*Este pacote de trabalho foca no treinamento dos modelos, na sua avaliação rigorosa e na validação da robustez dos resultados.*

* **4.1 Treinamento do Modelo Baseline**  
  - [ ] 4.1.1 Instanciar e treinar um modelo de Regressão Logística com os dados de treino.  
  - [ ] 4.1.2 Realizar previsões no conjunto de teste.  
* **4.2 Treinamento do Modelo Principal (XGBoost)**  
  - [ ] 4.2.1 Instanciar um XGBClassifier.  
  - [ ] 4.2.2 Configurar o hiperparâmetro scale\_pos\_weight para lidar com o desbalanceamento de classe, se houver.  
  - [ ] 4.2.3 Treinar o modelo com os dados de treino.  
  - [ ] 4.2.4 Realizar previsões no conjunto de teste.  
* **4.3 Avaliação de Métricas de Desempenho (na divisão simples)**  
  - [ ] 4.3.1 Para ambos os modelos (Regressão Logística e XGBoost), calcular e analisar:  
        * Matriz de Confusão.  
        * Precisão (Precision).  
        * Revocação (Recall).  
        * F1-Score.  
* **4.4 Validação Robusta (Walk-Forward)**  
  - [ ] 4.4.1 Implementar a estrutura de validação walk-forward simplificada com 3 dobras sobre os dados de teste.  
  - [ ] 4.4.2 Para cada dobra, treinar o modelo XGBoost com todos os dados anteriores e testar no período seguinte.  
  - [ ] 4.4.3 Coletar as métricas de desempenho (F1-Score, Precisão, etc.) para cada dobra.  
  - [ ] 4.4.4 Calcular a média e o desvio padrão das métricas obtidas na validação walk-forward.

### **5.0 \- Fase 5: Análise, Conclusão e Relatório (Dia 5\)**

*Este pacote de trabalho finaliza o projeto, consolidando os resultados, documentando as conclusões e identificando os próximos passos.*

* **5.1 Análise de Resultados e Overfitting**  
  - [ ] 5.1.1 Criar uma tabela comparativa com as métricas de desempenho de todos os modelos avaliados.  
  - [ ] 5.1.2 Analisar as pontuações de importância dos atributos (feature importance) do modelo XGBoost.  
  - [ ] 5.1.3 Avaliar os resultados da validação walk-forward para confirmar a robustez do modelo.  
  - [ ] 5.1.4 Documentar as estratégias de mitigação de overfitting utilizadas (validação cronológica, regularização do XGBoost).  
* **5.2 Documentação de Limitações e Próximos Passos**  
  - [ ] 5.2.1 Redigir uma seção sobre as limitações do modelo (Hipótese do Mercado Eficiente, ausência de dados alternativos).  
  - [ ] 5.2.2 Listar e detalhar os próximos passos recomendados (ajuste fino de hiperparâmetros, exploração de modelos LSTM/GRU, etc.).  
* **5.3 Preparação do Relatório Final**  
  - [ ] 5.3.1 Organizar o código-fonte (Jupyter Notebook) com comentários claros e células de Markdown explicativas.  
  - [ ] 5.3.2 Escrever um resumo executivo com os principais achados, o modelo campeão e seu desempenho final.  
  - [ ] 5.3.3 Arquivar todos os artefatos do projeto (código, dataset final, relatório).