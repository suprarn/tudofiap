## **Introdução: Enquadrando o Desafio da Previsão de Mercado**

A previsão de movimentos em mercados financeiros é uma das tarefas mais desafiadoras no campo da ciência de dados e finanças quantitativas. A Hipótese do Mercado Eficiente (HME), em sua forma fraca, postula que todos os preços históricos já estão refletidos no preço atual de um ativo, tornando a análise de dados passados, por si só, insuficiente para gerar retornos anormais de forma consistente.1 Este guia técnico parte do reconhecimento dessa realidade. O objetivo não é construir um oráculo infalível, mas sim desenvolver um modelo de machine learning que forneça uma vantagem estatística, mesmo que marginal, na previsão da direção diária do Índice Bovespa (IBOVESPA).

Para tornar este desafio tratável, o problema é reformulado. Em vez de tentar prever o valor exato do fechamento do dia seguinte — uma tarefa de regressão de altíssima complexidade e propensa a grandes erros —, o foco é transformado em um problema de classificação binária: o fechamento de amanhã será maior ou menor que o de hoje? Esta abordagem simplifica o espaço de saída e alinha-se melhor com decisões de negociação direcionais (compra ou venda). O resultado do modelo não será uma previsão determinística, mas sim uma probabilidade associada a cada classe ("alta" ou "baixa"), permitindo uma tomada de decisão mais nuançada e baseada em risco.3

É fundamental estabelecer expectativas realistas desde o início. A natureza caótica, dinâmica e não linear dos mercados financeiros significa que a previsão perfeita é uma impossibilidade teórica e prática.2 Um modelo que demonstre um desempenho consistentemente superior ao acaso (por exemplo, uma acurácia de 55% com um F1-score equilibrado em dados fora da amostra) já pode ser considerado valioso e potencialmente explorável em uma estratégia de negociação quantitativa.

---

## **Seção 1: Integridade dos Dados e Análise Exploratória**

A fase inicial de qualquer projeto de modelagem é a mais crítica. Erros ou mal-entendidos nesta etapa se propagarão por todo o fluxo de trabalho, comprometendo a validade dos resultados finais. O objetivo deste primeiro dia é garantir a qualidade impecável dos dados e desenvolver uma compreensão profunda de suas propriedades estatísticas.

### **1.1. Validação e Limpeza da Série Temporal**

A primeira ação consiste no carregamento do dataset de 15 anos de dados diários do IBOVESPA e na realização de uma verificação de integridade. Isso inclui confirmar o intervalo de datas, inspecionar a presença de timestamps duplicados e, mais importante, identificar lacunas na série temporal.

**Tratamento de Dados Ausentes (Dias Sem Pregão)**

Em um dataset de índice de ações diário, a principal causa de dados ausentes serão os fins de semana e feriados, dias em que a bolsa não opera.7 Essa ausência de dados não é aleatória, mas sim informativa e sistemática. A estratégia de tratamento deve refletir a mecânica do mercado. A abordagem mais adequada e tecnicamente correta neste cenário é o preenchimento para a frente (forward-fill, ou

ffill). A lógica subjacente é que, quando um mercado está fechado, seu valor não muda; ele permanece o mesmo do último fechamento de pregão.8 Portanto, usar

ffill não é uma *imputação* ou *estimativa* de um valor perdido, mas sim a *representação correta* do valor do índice durante um dia não útil. Métodos mais complexos, como a interpolação linear, seriam incorretos, pois criariam a falsa impressão de um movimento de preço suave em um dia em que nenhuma negociação ocorreu, introduzindo ruído artificial nos dados.9

**Tratamento de Outliers**

Séries financeiras, especialmente de mercados emergentes, podem exibir movimentos extremos. No entanto, em um índice diversificado como o IBOVESPA, a maioria dos "outliers" aparentes são, na verdade, reações legítimas do mercado a eventos macroeconômicos ou políticos significativos (por exemplo, crises financeiras, instabilidade política). Remover esses pontos seria remover informações valiosas sobre a volatilidade e o comportamento do mercado em momentos de estresse.7 A decisão estratégica para este projeto é não remover outliers. Em vez disso, a escolha recairá sobre modelos que são inerentemente robustos a eles, como os baseados em árvores de decisão (por exemplo, Random Forest e XGBoost).

### **1.2. Caracterizando o IBOVESPA: Análise Exploratória de Dados (EDA)**

Com os dados limpos e completos, a próxima etapa é gerar visualizações e estatísticas para entender o comportamento da série.

* **Gráfico de Preço e Volume:** A visualização primária deve ser um gráfico de linha do preço de fechamento (Close) ao longo dos 15 anos, para observar as tendências de longo prazo (bull markets, bear markets, períodos de lateralização). Abaixo deste, um gráfico de barras do volume diário ajuda a identificar períodos de alta e baixa atividade do mercado, que frequentemente se correlacionam com a volatilidade.10  
* **Análise de Retornos:** O foco da análise de séries temporais financeiras raramente está nos preços absolutos, mas sim em suas variações. É essencial calcular os retornos diários. Os retornos logarítmicos, calculados como log(Closet​/Closet−1​), são preferíveis por suas propriedades estatísticas (aditividade temporal). Um histograma da distribuição dos retornos diários é crucial. A expectativa é encontrar uma distribuição que se assemelha a uma normal, mas com "caudas gordas" (leptocurtose), indicando que eventos extremos (grandes altas ou quedas) são mais frequentes do que uma distribuição gaussiana preveria.  
* **Agrupamento de Volatilidade (Volatility Clustering):** Um gráfico de linha dos retornos diários ao longo do tempo revelará um dos fenômenos mais conhecidos dos mercados financeiros: o agrupamento de volatilidade. Períodos de grandes variações (alta volatilidade) tendem a ser seguidos por mais dias de alta volatilidade, enquanto períodos de calmaria tendem a ser seguidos por mais dias de calmaria.1 A identificação visual deste padrão justifica a inclusão de features baseadas em volatilidade na etapa de engenharia de atributos e reforça a natureza não-constante da variância do mercado.

### **1.3. O Teste Crítico de Estacionariedade**

O conceito de estacionariedade é fundamental na modelagem de séries temporais. Uma série é considerada estacionária se suas propriedades estatísticas — como média, variância e autocorrelação — são constantes ao longo do tempo.1 A maioria dos modelos clássicos e muitos algoritmos de machine learning assumem que os dados de entrada são estacionários. Modelar dados não estacionários pode levar a regressões espúrias, onde correlações falsas são identificadas, resultando em um modelo sem poder preditivo real.13

A validação da estacionariedade é, portanto, um passo obrigatório.

* **Ação e Justificativa:**  
  1. **Teste na Série de Preços:** Aplicar o teste Augmented Dickey-Fuller (ADF) e o teste Kwiatkowski-Phillips-Schmidt-Shin (KPSS) na série de preços de fechamento (Close).13 O teste ADF tem como hipótese nula a não estacionariedade (presença de raiz unitária), enquanto o KPSS tem como hipótese nula a estacionariedade. O uso de ambos os testes fornece uma conclusão mais robusta sobre a natureza da série.13 É esperado que a série de preços seja não estacionária, falhando em rejeitar a hipótese nula do ADF e rejeitando a do KPSS, devido à sua clara tendência de longo prazo.  
  2. **Teste na Série de Retornos:** Aplicar os mesmos testes ADF e KPSS na série de retornos diários. A expectativa é que esta série seja estacionária, pois a diferenciação (cálculo de retornos) é uma técnica comum para remover tendências e alcançar a estacionariedade.13

A confirmação de que os preços são não estacionários enquanto os retornos são estacionários é talvez o resultado mais importante de toda a fase de EDA. Esta constatação impõe uma restrição fundamental a todo o processo de modelagem subsequente: o modelo não deve aprender a partir dos *níveis* de preço, mas sim a partir das *mudanças* de preço (retornos). Qualquer atributo criado na Seção 3 deve, portanto, ser uma transformação estacionária dos dados originais, como os próprios retornos ou indicadores técnicos baseados neles. Ignorar este passo e usar preços brutos como features levaria quase certamente à construção de um modelo espúrio e inútil.13

---

## **Seção 2: Definição do Alvo de Previsão**

Com os dados validados e compreendidos, o problema de negócio deve ser traduzido em uma tarefa de machine learning clara e inequívoca. Isso envolve a criação da variável alvo (target) e a análise de suas características.

### **2.1. De Preço para Direção: Criando o Alvo Binário**

O objetivo é prever a direção do movimento do mercado no dia seguinte.

* **Lógica:** A meta é prever se o preço de fechamento do dia seguinte (Close(t+1)) será maior ou menor que o preço de fechamento do dia atual (t).  
* **Implementação:** Uma nova coluna, Target, será criada no dataframe. A lógica para cada dia t é a seguinte:  
  * Target(t) \= 1 se Close(t+1) \> Close(t) (dia de alta)  
  * Target(t) \= 0 se Close(t+1) \<= Close(t) (dia de baixa ou estável)  
* **Justificativa:** Esta formulação transforma um problema de regressão complexo (prever o valor exato) em um problema de classificação binária, que é geralmente mais tratável e menos sensível a erros de magnitude.15 A decisão de agrupar dias estáveis (  
  Close(t+1) \= Close(t)) com dias de baixa é uma escolha conservadora, tratando qualquer resultado que não seja uma alta como a classe "não-alta".  
* **Evitando o Viés de Lookahead (Lookahead Bias):** Este é um ponto crítico e uma fonte comum de erro. O alvo para o dia t (Target(t)) é calculado usando informações do dia t+1 (Close(t+1)). Portanto, ao construir o conjunto de features para o dia t, é imperativo que nenhuma informação do dia t+1 seja utilizada. Na prática, após calcular o Target para toda a série, a coluna deve ser deslocada (shift) um dia para trás, de modo que a linha do dia t contenha os atributos calculados até o fechamento de t e o alvo correspondente ao resultado do dia t+1.

### **2.2. Análise da Distribuição de Classes**

Após a criação da variável alvo, é essencial analisar sua distribuição.

* **Ação:** Calcular a frequência de 1s (altas) e 0s (baixas/estáveis) na coluna Target.  
* **Resultado Esperado:** Em um período de 15 anos que inclui mercados de alta significativos, é provável que as classes sejam ligeiramente desbalanceadas. Por exemplo, pode-se encontrar uma distribuição de 53% de dias de alta contra 47% de dias de baixa.  
* **Implicações:** A identificação deste desbalanceamento, mesmo que leve, tem consequências diretas e causais para as etapas posteriores do projeto.  
  1. **Seleção de Métricas:** O desbalanceamento invalida a acurácia como uma métrica de avaliação confiável. Um modelo que previsse sempre a classe majoritária (por exemplo, "alta") poderia atingir uma acurácia de 53%, parecendo útil, mas sem ter nenhum poder preditivo real.17 Isso torna obrigatório o uso de métricas mais robustas, como Precisão (Precision), Revocação (Recall) e F1-Score, que avaliam o desempenho em cada classe separadamente.17  
  2. **Treinamento do Modelo:** O desbalanceamento deve ser levado em conta durante o treinamento. Alguns algoritmos, como o XGBoost, possuem hiperparâmetros específicos para lidar com isso, como scale\_pos\_weight, que atribui um peso maior aos erros na classe minoritária.19 Alternativamente, poderiam ser usadas técnicas de reamostragem como SMOTE (oversampling) ou RandomUnderSampler (undersampling).18 No entanto, a implementação correta dessas técnicas em dados de séries temporais (evitando vazamento de dados entre folds de validação) é complexa. Portanto, a abordagem mais pragmática e segura é utilizar a ponderação de classes nativa do modelo. A simples etapa de criar e analisar a variável alvo já define requisitos técnicos para as seções de modelagem e avaliação.

---

## **Seção 3: Engenharia de Atributos Preditivos**

Esta é a etapa mais criativa e impactante do processo de modelagem. O objetivo é transformar os dados brutos de preço e volume em um conjunto rico de atributos (features) que capturem informações sobre a tendência recente, momento, volatilidade e psicologia do mercado. Todos os cálculos devem ser realizados com extremo cuidado para evitar o viés de lookahead.

### **3.1. Atributos Fundamentais: Lags e Retornos**

A informação mais direta sobre o futuro imediato é o passado imediato.

* **Ação:** Criar versões defasadas (lagged) dos retornos diários.  
* **Exemplo:** Retorno\_Lag\_1 (retorno de ontem), Retorno\_Lag\_2 (retorno de anteontem), e assim por diante, para um período de até 5 dias.  
* **Justificativa:** Retornos defasados são a forma mais simples e direta de fornecer ao modelo informações sobre o momento de curto prazo e a autocorrelação da série, permitindo que ele identifique padrões como "uma alta forte ontem aumenta a probabilidade de uma pequena queda hoje".21

### **3.2. Incorporando a Análise Técnica**

A análise técnica é um vasto campo dedicado a extrair sinais preditivos de dados de mercado. A engenharia de atributos pode se beneficiar enormemente da implementação de indicadores técnicos consagrados.

* **Ação:** Utilizar uma biblioteca especializada como pandas-ta para calcular de forma eficiente um conjunto curado e diversificado de indicadores.23 A seleção não deve ser exaustiva, mas sim representativa de diferentes dinâmicas de mercado.  
* **Justificativa:** Indicadores técnicos são transformações matemáticas de dados de preço e volume que condensam informações complexas em valores normalizados, representando a psicologia e a dinâmica do mercado.26 Eles servem como features de alta qualidade para os modelos de machine learning.  
* **Conjunto de Indicadores Proposto:**  
  * **Indicadores de Tendência (Médias Móveis Simples \- SMA):**  
    * SMA\_10, SMA\_20, SMA\_50: Representam as tendências de curto, médio e longo prazo.  
    * **Atributos Derivados:** Em vez de usar os valores brutos das SMAs (que seriam não estacionários), os atributos devem ser normalizados. Por exemplo: a razão entre o preço de fechamento e a SMA (Close / SMA\_20), que indica o quão "esticado" o preço está em relação à sua média recente; e a razão entre uma média curta e uma longa (SMA\_10 / SMA\_50), que quantifica sinais de cruzamento de médias (como o "golden cross" ou "death cross").25  
  * **Indicador de Momento (Índice de Força Relativa \- RSI):**  
    * RSI\_14: Mede a velocidade e a magnitude das mudanças de preço em uma escala de 0 a 100\. É um oscilador que ajuda a identificar condições de sobrecompra (tipicamente \> 70\) ou sobrevenda (tipicamente \< 30).25  
  * **Indicador de Volatilidade (Bandas de Bollinger):**  
    * BBands\_20: Consiste em uma SMA de 20 dias (banda do meio) e bandas superior e inferior, que são tipicamente dois desvios padrão acima e abaixo da banda do meio.  
    * **Atributos Derivados:** A largura das bandas, normalizada pela banda do meio ((Banda\_Superior \- Banda\_Inferior) / Banda\_do\_Meio), é um excelente indicador de volatilidade. A posição do preço de fechamento dentro das bandas ((Close \- Banda\_Inferior) / (Banda\_Superior \- Banda\_Inferior)) normaliza o preço em relação à sua faixa de volatilidade recente.  
  * **Indicador de Volume (On-Balance Volume \- OBV):**  
    * Calcula um total acumulado de volume, somando o volume em dias de alta e subtraindo em dias de baixa. O OBV pode confirmar a força de um movimento de preço ou sinalizar uma divergência.

### **3.3. Evitando o Viés de Lookahead no Cálculo dos Atributos**

A regra de ouro da modelagem de séries temporais deve ser seguida rigorosamente: o conjunto de atributos para prever o resultado do dia t+1 só pode usar informações disponíveis até o fechamento do pregão do dia t.

* **Exemplo Prático:** Para calcular a SMA\_10 para o dia t, a média é calculada usando os preços de fechamento do dia t-9 até o dia t, inclusive. Este valor é então usado como um atributo para prever o Target(t), que por sua vez é baseado no Close(t+1). Um erro comum é usar uma média móvel centrada, que utilizaria dados futuros (t+1, t+2, etc.) para calcular o valor em t, o que invalidaria completamente o modelo.22

### **Tabela 1: Dicionário de Engenharia de Atributos**

Para garantir clareza, reprodutibilidade e documentação, é fundamental criar uma tabela que sirva como uma fonte única de verdade para todos os atributos projetados.

| Nome do Atributo | Cálculo | Dados de Entrada | Intuição Financeira | Categoria |
| :---- | :---- | :---- | :---- | :---- |
| Return\_Lag\_1 | Retorno do dia anterior | Close | Inércia/reversão de curto prazo | Momento |
| Return\_Lag\_5 | Retorno de 5 dias atrás | Close | Inércia/reversão de médio prazo | Momento |
| Ratio\_Close\_SMA20 | Close / SMA\_20 | Close | Mede o quão "esticado" o preço está em relação à sua média de curto prazo | Tendência |
| Ratio\_SMA10\_SMA50 | SMA\_10 / SMA\_50 | Close | Sinaliza cruzamentos de médias, indicando mudanças de tendência | Tendência |
| RSI\_14 | Índice de Força Relativa de 14 dias | Close | Mede o momento; valores \> 70 sugerem sobrecompra, \< 30 sobrevenda | Momento |
| BB\_Width\_20 | (BB\_Upper \- BB\_Lower) / BB\_Middle | Close | Mede a volatilidade do mercado; valores altos indicam alta volatilidade | Volatilidade |
| BB\_Position\_20 | (Close \- BB\_Lower) / (BB\_Upper \- BB\_Lower) | Close | Posição normalizada do preço dentro da sua faixa de volatilidade recente | Volatilidade |
| OBV\_pct\_change | Variação percentual do OBV | Close, Volume | Mede a pressão de compra/venda baseada no volume | Volume |

---

## **Seção 4: Preparação da Base de Dados Temporalmente Consciente**

Esta seção constrói a ponte entre os dados brutos da série temporal e o formato estruturado (tabular) exigido pelos algoritmos de aprendizado supervisionado.

### **4.1. O Método da Janela Deslizante (Sliding Window)**

Esta é a técnica padrão para transformar uma série temporal em um dataset de aprendizado supervisionado.32

* **Conceito:** O método envolve a criação de "janelas" de observações passadas como atributos de entrada para prever um alvo futuro.  
* **Ação:** Definir um tamanho de janela de entrada (por exemplo, n=5 dias). Para cada dia t no dataset, os atributos do modelo serão o conjunto de todos os indicadores da Tabela 1 calculados para os dias t-4, t-3, t-2, t-1 e t. O alvo para esta amostra será o valor de Target(t) (que corresponde ao movimento de t+1). Este processo é repetido para cada dia da série, criando um grande dataset tabular onde cada linha representa uma janela de 5 dias de atributos e seu correspondente resultado.  
* **Justificativa:** Modelos não sequenciais como Regressão Logística e XGBoost não compreendem inerentemente a ordem temporal. Esta estrutura de janela fornece explicitamente ao modelo um "instantâneo" do passado recente para que ele possa aprender as relações entre os padrões recentes e o resultado futuro.34 Para modelos sequenciais como LSTM, esta estrutura também é o formato de entrada padrão (uma sequência de vetores de atributos).

### **4.2. Divisão Cronológica dos Dados: A Regra Inegociável**

A divisão do dataset em conjuntos de treino e teste é talvez o passo mais crítico para garantir a validade do modelo.

* **Ação:** Dividir o dataset estruturado (após a aplicação da janela deslizante) em um conjunto de treino e um conjunto de teste com base em uma data de corte fixa. **A divisão aleatória (train\_test\_split com shuffle=True) nunca deve ser usada.**  
* **Exemplo:** Com 15 anos de dados (\~3750 dias de pregão), uma divisão 80-20 é apropriada.  
  * **Conjunto de Treino:** Os primeiros 12 anos de dados (aproximadamente 3000 dias).  
  * **Conjunto de Teste:** Os 3 anos de dados mais recentes (aproximadamente 750 dias).  
* **Justificativa:** Esta é a única maneira de prevenir o vazamento de dados do futuro para o passado e de simular realisticamente como o modelo se comportaria em um ambiente de produção. O modelo deve ser treinado exclusivamente com dados passados e testado em um "futuro" completamente desconhecido (o conjunto de teste).36 A divisão aleatória permitiria que o modelo "visse o futuro" durante o treinamento, aprendendo padrões do período de teste, o que levaria a métricas de desempenho inflacionadas e totalmente irrealistas. A divisão cronológica transforma a avaliação do modelo de um simples exercício de medição de erro em uma simulação de desempenho histórico, tornando os resultados muito mais rigorosos e confiáveis.

### **4.3. Escalonamento de Atributos (Feature Scaling)**

* **Ação:** Ajustar (fit) um StandardScaler do scikit-learn *apenas nos dados de treinamento*. Em seguida, usar este scaler já ajustado para transformar (transform) tanto os dados de treinamento quanto os de teste.  
* **Justificativa:** Muitos algoritmos, especialmente os lineares (Regressão Logística) e as redes neurais, funcionam melhor quando os atributos estão em uma escala semelhante. O StandardScaler normaliza os atributos removendo a média e escalonando para a variância unitária.15 É de vital importância ajustar o scaler  
  *apenas* nos dados de treino para evitar que qualquer informação do conjunto de teste (como sua média e desvio padrão) "vaze" para o processo de treinamento. Embora modelos baseados em árvores como XGBoost sejam menos sensíveis ao escalonamento, aplicá-lo é uma boa prática que não prejudica o desempenho e garante a consistência do pipeline.

---

## **Seção 5: Uma Abordagem Pragmática para a Seleção de Modelos**

A escolha do modelo é um compromisso entre desempenho, complexidade, interpretabilidade e tempo de desenvolvimento.

### **5.1. A Linha de Base: Regressão Logística**

* **Ação:** Treinar um modelo simples de Regressão Logística.  
* **Justificativa:** É um modelo rápido de treinar, seus coeficientes são interpretáveis (fornecendo uma visão sobre as relações lineares entre os atributos e o alvo) e, mais importante, estabelece uma linha de base de desempenho crucial. Qualquer modelo mais complexo deve superar significativamente a Regressão Logística para justificar sua complexidade adicional.9 Se um modelo sofisticado como o XGBoost não conseguir um desempenho melhor, isso sugere que as relações preditivas nos dados são fracas ou não lineares de uma forma que o modelo de árvore possa capturar.

### **5.2. O Cavalo de Batalha: XGBoost (eXtreme Gradient Boosting)**

* **Ação:** Dedicar a maior parte do tempo de modelagem à implementação e ajuste de um classificador XGBoost.  
* **Justificativa:**  
  * **Desempenho:** XGBoost é um método de ensemble de última geração que consistentemente apresenta desempenho de ponta para dados tabulares estruturados, que é exatamente o formato que criamos.39  
  * **Velocidade e Eficiência:** É altamente otimizado para desempenho e pode ser treinado muito mais rapidamente do que modelos de deep learning, tornando-o ideal para um projeto com restrições de tempo.40  
  * **Robustez:** Como um ensemble de árvores de decisão, ele lida bem com interações complexas entre os atributos e é robusto a outliers.  
  * **Interpretabilidade:** O XGBoost fornece pontuações de importância de atributos (feature importance), permitindo uma análise de quais indicadores são mais influentes nas previsões do modelo. Esta é uma vantagem significativa sobre modelos "caixa-preta".40

### **5.3. Os Especialistas Sequenciais: LSTM & GRU (Para Iteração Futura)**

* **Discussão:** É importante discutir as redes Long Short-Term Memory (LSTM) e Gated Recurrent Unit (GRU) como alternativas.  
* **Pontos Fortes:** Estas são redes neurais recorrentes (RNNs) projetadas especificamente para capturar dependências de longo prazo em dados sequenciais. Sua principal vantagem teórica sobre modelos baseados em árvores é a capacidade de aprender a partir da ordem dos dados de forma nativa.42  
* **Pontos Fracos (no contexto deste projeto):** São significativamente mais complexos de construir e ajustar (exigindo decisões sobre número de camadas, unidades por camada, funções de ativação, otimizadores, etc.), demandam tempos de treinamento muito mais longos e são inerentemente menos interpretáveis ("caixas-pretas").40  


### **Tabela 2: Análise Comparativa de Modelos**

Para justificar de forma clara e concisa a escolha do XGBoost, uma tabela comparativa é a ferramenta ideal. Ela permite uma avaliação lado a lado dos candidatos com base nos critérios mais relevantes para o projeto.

| Modelo | Tipo de Modelo | Desempenho Típico | Interpretabilidade | Lida com Sequências Nativamente? |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Regressão Logística** | Linear | Baixo a Médio | Baixo | Alta | Não |
| **XGBoost** | Ensemble de Árvores | Alto | Médio | Média (Importância dos Atributos) | Não |
| **LSTM / GRU** | Rede Neural Recorrente | Alto a Muito Alto | Alto | Baixa | Sim |

Esta tabela demonstra visualmente por que o XGBoost representa o melhor equilíbrio para este projeto: oferece alto desempenho com tempo de desenvolvimento manejável e um grau de interpretabilidade útil, ao contrário da alta complexidade e opacidade dos modelos de deep learning.

---

## **Seção 6: Validação Robusta do Modelo e Métricas de Desempenho**

A avaliação deve ser rigorosa, utilizando métricas apropriadas para um problema de classificação potencialmente desbalanceado e um esquema de validação que respeite a natureza temporal dos dados.

### **6.1. Métricas que Importam: Além da Acurácia**

* **Ação:** Avaliar o modelo usando um conjunto de métricas derivadas da matriz de confusão.  
* **Justificativa:** Conforme estabelecido na Seção 2, as classes são provavelmente desbalanceadas, tornando a acurácia uma métrica pobre e enganosa.18 É necessário entender o comportamento do modelo em suas previsões de "alta" e "baixa" separadamente.  
* Métricas Chave 17:  
  * **Matriz de Confusão:** A tabela 2x2 que exibe os Verdadeiros Positivos (TP), Falsos Positivos (FP), Verdadeiros Negativos (TN) e Falsos Negativos (FN). É a fonte de todas as outras métricas e fornece a visão mais granular do desempenho.  
  * **Precisão (Precision):** Para a classe "alta" (1), é calculada como TP/(TP+FP). Responde à pergunta: "Quando o modelo prevê que o mercado vai subir, qual a porcentagem de vezes que ele está correto?". Esta métrica está diretamente ligada à confiabilidade de um sinal de compra. Uma alta precisão significa poucos alarmes falsos.  
  * **Revocação (Recall / Sensibilidade):** Para a classe "alta", é calculada como TP/(TP+FN). Responde à pergunta: "De todos os dias em que o mercado realmente subiu, qual a porcentagem que o modelo identificou corretamente?". Esta métrica está ligada à capacidade do modelo de capturar oportunidades.  
  * **F1-Score:** A média harmônica da precisão e da revocação, calculada como 2×(Precisa\~o×Revocac\\c​a\~o)/(Precisa\~o+Revocac\\c​a\~o). Fornece uma pontuação única e equilibrada, sendo extremamente útil quando há uma distribuição de classes desigual, pois penaliza modelos que são bons em uma métrica à custa da outra.18

A escolha entre otimizar para precisão ou para revocação não é apenas uma decisão técnica, mas reflete um objetivo estratégico. Um modelo otimizado para precisão fará menos previsões de "alta", mas as que fizer serão muito confiáveis, alinhando-se a uma estratégia de negociação de baixa frequência e alta confiança. Em contraste, um modelo otimizado para revocação tentará capturar todos os dias de alta possíveis, mesmo que isso signifique gerar mais falsos positivos, alinhando-se a uma estratégia de maior frequência e busca por oportunidades. O F1-Score oferece um meio-termo equilibrado.

### **6.2. Simulando a Realidade: Validação Walk-Forward**

Uma única divisão treino-teste, embora cronológica, pode ser suscetível à sorte; o período de teste pode ser atipicamente fácil ou difícil de prever. Uma validação mais robusta é necessária.

* **Conceito:** A validação walk-forward (ou validação em janela rolante) é uma técnica de validação cruzada específica para séries temporais que simula melhor uma estratégia de negociação real. Em vez de uma única divisão, ela usa uma série de divisões de treino/teste que "caminham" através do tempo.34  
* **Ação:** Implementar uma validação walk-forward simplificada com algumas dobras (folds) sobre o conjunto de dados de teste. Por exemplo, se os últimos 3 anos foram reservados para teste:  
  * **Dobra 1:** Treinar com dados dos anos 1-12, testar no ano 13\.  
  * **Dobra 2:** Treinar com dados dos anos 1-13, testar no ano 14\.  
  * **Dobra 3:** Treinar com dados dos anos 1-14, testar no ano 15\.  
* **Justificativa:** Esta abordagem testa a robustez do modelo em diferentes regimes de mercado (por exemplo, um ano de alta, um de baixa, um lateral).51 Um modelo que se sai bem em apenas um período de teste pode ser um acaso. Um desempenho consistente em múltiplas dobras walk-forward fornece uma confiança muito maior na capacidade de generalização do modelo.49 O desempenho final do modelo é relatado como a média (e o desvio padrão) das métricas em todas as dobras de teste.

---

## **Seção 7: Overfitting, Trade-offs e Considerações Finais**

A seção final aborda o principal risco na modelagem financeira, resume as escolhas pragmáticas feitas para cumprir o prazo e delineia um caminho para o futuro.

### **7.1. Domando o Overfitting**

* **Conceito:** O overfitting ocorre quando um modelo aprende o ruído nos dados de treinamento em vez do sinal subjacente. Isso resulta em um desempenho excelente nos dados vistos, mas um desempenho pobre e não confiável em dados novos e não vistos.53 É o pecado capital da modelagem financeira.  
* **Estratégias de Defesa Empregadas:**  
  1. **Validação Robusta:** O uso da divisão cronológica e da validação walk-forward é a principal linha de defesa. Garante que o modelo seja sempre avaliado em dados futuros não vistos, simulando a realidade.34  
  2. **Regularização:** O XGBoost possui hiperparâmetros de regularização L1 (alpha) e L2 (lambda) que penalizam a complexidade do modelo (por exemplo, o tamanho dos pesos das folhas), ajudando a evitar que ele se ajuste ao ruído.54 O ajuste desses parâmetros é uma parte importante da otimização do modelo.  
  3. **Seleção de Atributos:** A utilização de um conjunto curado e diversificado de atributos, em vez de centenas de indicadores correlacionados, reduz o risco de o modelo encontrar padrões espúrios e superajustar-se a eles.54

### **7.2. A Arte do Compromisso**

* **Resumo das Decisões Pragmáticas:**  
  * **Tratamento de Dados:** Escolha do método ffill, simples e robusto, em vez de imputações complexas.  
  * **Engenharia de Atributos:** Seleção de um conjunto pequeno e diversificado de indicadores em vez de uma lista exaustiva.  
  * **Seleção de Modelo:** Priorização do XGBoost por seu equilíbrio entre velocidade e desempenho, em detrimento dos mais complexos LSTM/GRU.  
  * **Validação:** Recomendação de uma validação walk-forward simplificada, viável dentro do cronograma.  
* **Justificativa:** Todas essas decisões foram tomadas para mitigar os riscos do projeto e garantir que um modelo de ponta a ponta, viável e metodologicamente correto, seja produzido em cinco dias, sem comprometer os princípios fundamentais, como evitar o viés de lookahead e usar validação temporalmente apropriada.38

### **7.3. Reconhecendo Limitações e Planejando o Futuro**

* **Eficiência do Mercado:** É crucial reiterar que nenhum modelo pode prever o mercado com perfeição. O objetivo é sempre buscar uma vantagem estatística.


#### **Referências citadas**

1. 9.3 Time series analysis \- Financial Mathematics \- Fiveable, acessado em julho 24, 2025, [https://library.fiveable.me/financial-mathematics/unit-9/time-series-analysis/study-guide/KnPZlCw6XjYuunHf](https://library.fiveable.me/financial-mathematics/unit-9/time-series-analysis/study-guide/KnPZlCw6XjYuunHf)  
2. (PDF) Stock Market Prediction Using Machine Learning Techniques: A Decade Survey on Methodologies, Recent Developments, and Future Directions \- ResearchGate, acessado em julho 24, 2025, [https://www.researchgate.net/publication/356008402\_Stock\_Market\_Prediction\_Using\_Machine\_Learning\_Techniques\_A\_Decade\_Survey\_on\_Methodologies\_Recent\_Developments\_and\_Future\_Directions](https://www.researchgate.net/publication/356008402_Stock_Market_Prediction_Using_Machine_Learning_Techniques_A_Decade_Survey_on_Methodologies_Recent_Developments_and_Future_Directions)  
3. (PDF) Advancements in Financial Market Predictions Using Machine Learning Techniques, acessado em julho 24, 2025, [https://www.researchgate.net/publication/382315352\_Advancements\_in\_Financial\_Market\_Predictions\_Using\_Machine\_Learning\_Techniques](https://www.researchgate.net/publication/382315352_Advancements_in_Financial_Market_Predictions_Using_Machine_Learning_Techniques)  
4. Advancements in Financial Market Predictions Using Machine Learning Techniques, acessado em julho 24, 2025, [https://www.preprints.org/manuscript/202407.1075/v1](https://www.preprints.org/manuscript/202407.1075/v1)  
5. A Comparative Study of Machine Learning Algorithms for Stock Price Prediction Using Insider Trading Data \- arXiv, acessado em julho 24, 2025, [https://arxiv.org/html/2502.08728v1](https://arxiv.org/html/2502.08728v1)  
6. Machine Learning in Stock Price Prediction: A Review of Techniques and Challenges \- ASPG, acessado em julho 24, 2025, [https://www.americaspg.com/article/pdf/3409](https://www.americaspg.com/article/pdf/3409)  
7. 12.9 Dealing with missing values and outliers | Forecasting ... \- OTexts, acessado em julho 24, 2025, [https://otexts.com/fpp2/missing-outliers.html](https://otexts.com/fpp2/missing-outliers.html)  
8. Handling missing data (holidays) in multiple time series (historical simulation VaR), acessado em julho 24, 2025, [https://stats.stackexchange.com/questions/70480/handling-missing-data-holidays-in-multiple-time-series-historical-simulation](https://stats.stackexchange.com/questions/70480/handling-missing-data-holidays-in-multiple-time-series-historical-simulation)  
9. Common Challenges in Time Series Financial Forecasting \- Phoenix Strategy Group, acessado em julho 24, 2025, [https://www.phoenixstrategy.group/blog/common-challenges-in-time-series-financial-forecasting](https://www.phoenixstrategy.group/blog/common-challenges-in-time-series-financial-forecasting)  
10. A Guide to Time Series Analysis in Python | Built In, acessado em julho 24, 2025, [https://builtin.com/data-science/time-series-python](https://builtin.com/data-science/time-series-python)  
11. Understanding Time Series Analysis in Python \- Simplilearn.com, acessado em julho 24, 2025, [https://www.simplilearn.com/tutorials/python-tutorial/time-series-analysis-in-python](https://www.simplilearn.com/tutorials/python-tutorial/time-series-analysis-in-python)  
12. Evaluating Volatility Using an ANFIS Model for Financial Time Series Prediction \- MDPI, acessado em julho 24, 2025, [https://www.mdpi.com/2227-9091/12/10/156](https://www.mdpi.com/2227-9091/12/10/156)  
13. Mastering Stationarity in Time Series for Financial Data, acessado em julho 24, 2025, [https://www.numberanalytics.com/blog/mastering-stationarity-time-series-financial-data](https://www.numberanalytics.com/blog/mastering-stationarity-time-series-financial-data)  
14. Classification of Time Series with LSTM RNN \- Kaggle, acessado em julho 24, 2025, [https://www.kaggle.com/code/szaitseff/classification-of-time-series-with-lstm-rnn](https://www.kaggle.com/code/szaitseff/classification-of-time-series-with-lstm-rnn)  
15. Predicting Stock Direction (Binary Classification) \- Kaggle, acessado em julho 24, 2025, [https://www.kaggle.com/code/lutielle/predicting-stock-direction-binary-classification](https://www.kaggle.com/code/lutielle/predicting-stock-direction-binary-classification)  
16. Importance of Event Binary Features in Stock Price Prediction \- ResearchGate, acessado em julho 24, 2025, [https://www.researchgate.net/publication/339573587\_Importance\_of\_Event\_Binary\_Features\_in\_Stock\_Price\_Prediction](https://www.researchgate.net/publication/339573587_Importance_of_Event_Binary_Features_in_Stock_Price_Prediction)  
17. 20 Evaluation Metrics for Binary Classification \- Neptune.ai, acessado em julho 24, 2025, [https://neptune.ai/blog/evaluation-metrics-binary-classification](https://neptune.ai/blog/evaluation-metrics-binary-classification)  
18. 5 Effective Ways to Handle Imbalanced Data in Machine Learning, acessado em julho 24, 2025, [https://machinelearningmastery.com/5-effective-ways-to-handle-imbalanced-data-in-machine-learning/](https://machinelearningmastery.com/5-effective-ways-to-handle-imbalanced-data-in-machine-learning/)  
19. Comparative Analysis of Resampling Techniques for Class ... \- MDPI, acessado em julho 24, 2025, [https://www.mdpi.com/2227-7390/13/13/2186](https://www.mdpi.com/2227-7390/13/13/2186)  
20. The Ultimate Guide to Handling Class Imbalance with 11 Techniques: CRISP-DM Data Preparation | by Donato\_TH | Donato Story | Medium, acessado em julho 24, 2025, [https://medium.com/donato-story/the-ultimate-guide-to-handling-class-imbalance-with-11-techniques-crisp-dm-data-preparation-5d3c592d3f7b](https://medium.com/donato-story/the-ultimate-guide-to-handling-class-imbalance-with-11-techniques-crisp-dm-data-preparation-5d3c592d3f7b)  
21. Top Machine Learning Projects for Finance Students & Analysts, acessado em julho 24, 2025, [https://eicta.iitk.ac.in/knowledge-hub/machine-learning/top-machine-learning-projects-for-finance/](https://eicta.iitk.ac.in/knowledge-hub/machine-learning/top-machine-learning-projects-for-finance/)  
22. Implement Walk-Forward Optimization with XGBoost for Stock Price Prediction in Python, acessado em julho 24, 2025, [https://blog.quantinsti.com/walk-forward-optimization-python-xgboost-stock-prediction/](https://blog.quantinsti.com/walk-forward-optimization-python-xgboost-stock-prediction/)  
23. Predict Stock Prices Using Technical Indicators and Machine Learning in Python \- YouTube, acessado em julho 24, 2025, [https://www.youtube.com/watch?v=gtk8k8G-\_3k](https://www.youtube.com/watch?v=gtk8k8G-_3k)  
24. Technical Analysis Indicators \- Pandas TA is an easy to use Python 3 Pandas Extension with 130+ Indicators \- GitHub, acessado em julho 24, 2025, [https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas](https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas)  
25. Using Pandas\_TA to generate Technical Indicators and Signals \- pythonology, acessado em julho 24, 2025, [https://pythonology.eu/using-pandas\_ta-to-generate-technical-indicators-and-signals/](https://pythonology.eu/using-pandas_ta-to-generate-technical-indicators-and-signals/)  
26. (PDF) PREDICTING STOCK MARKET DIRECTION USING MACHINE LEARNING MODELS, acessado em julho 24, 2025, [https://www.researchgate.net/publication/363506105\_PREDICTING\_STOCK\_MARKET\_DIRECTION\_USING\_MACHINE\_LEARNING\_MODELS](https://www.researchgate.net/publication/363506105_PREDICTING_STOCK_MARKET_DIRECTION_USING_MACHINE_LEARNING_MODELS)  
27. A Survey on Machine Learning for Stock Price Prediction: Algorithms and Techniques \- ePrints Soton \- University of Southampton, acessado em julho 24, 2025, [https://eprints.soton.ac.uk/437785/1/FEMIB\_2020\_6.pdf](https://eprints.soton.ac.uk/437785/1/FEMIB_2020_6.pdf)  
28. How to Develop an AI Stock Prediction Software? A Complete Guide \- Matellio Inc, acessado em julho 24, 2025, [https://www.matellio.com/blog/ai-stock-prediction-software-development/](https://www.matellio.com/blog/ai-stock-prediction-software-development/)  
29. XGBoost for stock trend & prices prediction \- Kaggle, acessado em julho 24, 2025, [https://www.kaggle.com/code/mtszkw/xgboost-for-stock-trend-prices-prediction](https://www.kaggle.com/code/mtszkw/xgboost-for-stock-trend-prices-prediction)  
30. Implementing Technical Indicators in Python for Trading \- PyQuant News, acessado em julho 24, 2025, [https://www.pyquantnews.com/free-python-resources/implementing-technical-indicators-in-python-for-trading](https://www.pyquantnews.com/free-python-resources/implementing-technical-indicators-in-python-for-trading)  
31. Mid-price prediction based on machine learning methods with technical and quantitative indicators | PLOS One \- Research journals, acessado em julho 24, 2025, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234107](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234107)  
32. What is a sliding window approach in time series forecasting? \- Milvus, acessado em julho 24, 2025, [https://milvus.io/ai-quick-reference/what-is-a-sliding-window-approach-in-time-series-forecasting](https://milvus.io/ai-quick-reference/what-is-a-sliding-window-approach-in-time-series-forecasting)  
33. Sliding Window Technique — reduce the complexity of your algorithm | by Data Overload, acessado em julho 24, 2025, [https://medium.com/@data-overload/sliding-window-technique-reduce-the-complexity-of-your-algorithm-5badb2cf432f](https://medium.com/@data-overload/sliding-window-technique-reduce-the-complexity-of-your-algorithm-5badb2cf432f)  
34. How to Use XGBoost for Time Series Forecasting \- MachineLearningMastery.com, acessado em julho 24, 2025, [https://machinelearningmastery.com/xgboost-for-time-series-forecasting/](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)  
35. Random Forest for Time Series Forecasting \- MachineLearningMastery.com, acessado em julho 24, 2025, [https://machinelearningmastery.com/random-forest-for-time-series-forecasting/](https://machinelearningmastery.com/random-forest-for-time-series-forecasting/)  
36. Splitting Time Series Data into Train/Test/Validation Sets \- Cross ..., acessado em julho 24, 2025, [https://stats.stackexchange.com/questions/346907/splitting-time-series-data-into-train-test-validation-sets](https://stats.stackexchange.com/questions/346907/splitting-time-series-data-into-train-test-validation-sets)  
37. What are some best practices for splitting a dataset into training, validation, and test sets?, acessado em julho 24, 2025, [https://milvus.io/ai-quick-reference/what-are-some-best-practices-for-splitting-a-dataset-into-training-validation-and-test-sets](https://milvus.io/ai-quick-reference/what-are-some-best-practices-for-splitting-a-dataset-into-training-validation-and-test-sets)  
38. A Comparative Analysis of Machine Learning Models for Predictive Analytics in Finance \- ARIMSI, acessado em julho 24, 2025, [https://international.arimsi.or.id/index.php/IJAMC/article/download/3/3/259](https://international.arimsi.or.id/index.php/IJAMC/article/download/3/3/259)  
39. Comparative Analysis of Machine Learning Models in Financial Services \- ResearchGate, acessado em julho 24, 2025, [https://www.researchgate.net/publication/389653896\_Comparative\_Analysis\_of\_Machine\_Learning\_Models\_in\_Financial\_Services](https://www.researchgate.net/publication/389653896_Comparative_Analysis_of_Machine_Learning_Models_in_Financial_Services)  
40. Comparing XGBoost and LSTM Models for Prediction of Microsoft Corp's Stock Price Direction \- mtu-mujast, acessado em julho 24, 2025, [https://mujast.mtu.edu.ng/storage/issues/Year\_2024\_Vol\_4/Number\_2/1729800557\_MUJAST\_240801.pdf](https://mujast.mtu.edu.ng/storage/issues/Year_2024_Vol_4/Number_2/1729800557_MUJAST_240801.pdf)  
41. Comparing Machine Learning Methods—SVR, XGBoost, LSTM, and MLP— For Forecasting the Moroccan Stock Market \- MDPI, acessado em julho 24, 2025, [https://www.mdpi.com/2813-0324/7/1/39](https://www.mdpi.com/2813-0324/7/1/39)  
42. A Comparative Analysis of Machine Learning Models for ... \- ARIMSI, acessado em julho 24, 2025, [https://international.arimsi.or.id/index.php/IJAMC/article/download/71/52/225](https://international.arimsi.or.id/index.php/IJAMC/article/download/71/52/225)  
43. Introduction to XGBoost in Python \- QuantInsti Blog, acessado em julho 24, 2025, [https://blog.quantinsti.com/xgboost-python/](https://blog.quantinsti.com/xgboost-python/)  
44. Advanced Stock Market Prediction Using Hybrid GRU-LSTM Techniques \- ijariie, acessado em julho 24, 2025, [https://ijariie.com/AdminUploadPdf/TITLE\_\_Advanced\_Stock\_Market\_Prediction\_\_Using\_Hybrid\_GRU\_LSTM\_Techniques\_ijariie25797.pdf](https://ijariie.com/AdminUploadPdf/TITLE__Advanced_Stock_Market_Prediction__Using_Hybrid_GRU_LSTM_Techniques_ijariie25797.pdf)  
45. Stock Price Prediction using LSTM and its Implementation \- Analytics Vidhya, acessado em julho 24, 2025, [https://www.analyticsvidhya.com/blog/2021/12/stock-price-prediction-using-lstm/](https://www.analyticsvidhya.com/blog/2021/12/stock-price-prediction-using-lstm/)  
46. Comparison of XGBoost and LSTM Models for Stock Price Prediction \- ResearchGate, acessado em julho 24, 2025, [https://www.researchgate.net/publication/376887936\_Comparison\_of\_XGBoost\_and\_LSTM\_Models\_for\_Stock\_Price\_Prediction](https://www.researchgate.net/publication/376887936_Comparison_of_XGBoost_and_LSTM_Models_for_Stock_Price_Prediction)  
47. APPL stock price prediction based on LSTM and GRU \- Advances in Engineering Innovation, acessado em julho 24, 2025, [https://www.ewadirect.com/proceedings/ace/article/view/10967](https://www.ewadirect.com/proceedings/ace/article/view/10967)  
48. Evaluation of binary classifiers \- Wikipedia, acessado em julho 24, 2025, [https://en.wikipedia.org/wiki/Evaluation\_of\_binary\_classifiers](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)  
49. Time Series Cross-Validation \- GeeksforGeeks, acessado em julho 24, 2025, [https://www.geeksforgeeks.org/machine-learning/time-series-cross-validation/](https://www.geeksforgeeks.org/machine-learning/time-series-cross-validation/)  
50. Cross Validation in Time Series. Cross Validation: | by Soumya Shrivastava | Medium, acessado em julho 24, 2025, [https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4)  
51. Walk-Forward Optimization: How It Works, Its Limitations, and Backtesting Implementation, acessado em julho 24, 2025, [https://blog.quantinsti.com/walk-forward-optimization-introduction/](https://blog.quantinsti.com/walk-forward-optimization-introduction/)  
52. Walk forward validation : r/datascience \- Reddit, acessado em julho 24, 2025, [https://www.reddit.com/r/datascience/comments/18pxc6x/walk\_forward\_validation/](https://www.reddit.com/r/datascience/comments/18pxc6x/walk_forward_validation/)  
53. onemoneyway.com, acessado em julho 24, 2025, [https://onemoneyway.com/en/dictionary/overfitting/\#:\~:text=Overfitting%20occurs%20when%20a%20financial,model%20from%20identifying%20broader%20trends.](https://onemoneyway.com/en/dictionary/overfitting/#:~:text=Overfitting%20occurs%20when%20a%20financial,model%20from%20identifying%20broader%20trends.)  
54. Overfitting in finance: causes, detection & prevention strategies, acessado em julho 24, 2025, [https://onemoneyway.com/en/dictionary/overfitting/](https://onemoneyway.com/en/dictionary/overfitting/)  
55. 7.3 Overfitting and underfitting \- Intro To Time Series \- Fiveable, acessado em julho 24, 2025, [https://library.fiveable.me/intro-time-series/unit-7/overfitting-underfitting/study-guide/hkQQAu3x5Sk5OoHo](https://library.fiveable.me/intro-time-series/unit-7/overfitting-underfitting/study-guide/hkQQAu3x5Sk5OoHo)  
56. Stock Market Prediction Using Machine Learning and Deep Learning Techniques: A Review, acessado em julho 24, 2025, [https://www.mdpi.com/2673-9909/5/3/76](https://www.mdpi.com/2673-9909/5/3/76)  
57. Machine Learning Approaches in Stock Price Prediction: A Systematic Review, acessado em julho 24, 2025, [https://www.researchgate.net/publication/357763580\_Machine\_Learning\_Approaches\_in\_Stock\_Price\_Prediction\_A\_Systematic\_Review](https://www.researchgate.net/publication/357763580_Machine_Learning_Approaches_in_Stock_Price_Prediction_A_Systematic_Review)  
58. Machine Learning for Financial Forecasting, Planning and Analysis: Recent Developments and Pitfalls \- arXiv, acessado em julho 24, 2025, [https://arxiv.org/pdf/2107.04851](https://arxiv.org/pdf/2107.04851)