#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 1: Integridade dos Dados e Análise Exploratória (EAP 1.0)
==============================================================

Este módulo implementa a Fase 1 do projeto de previsão de tendência do IBOVESPA,
focando em garantir a qualidade dos dados brutos e extrair insights iniciais 
sobre o comportamento do mercado.

Autor: Projeto Tech Challenge 2
Data: 2025-01-24
Referência: EAP.md e Steering.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos em português
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

class IntegridadeDadosEDA:
    """
    Classe responsável pela validação, limpeza e análise exploratória 
    dos dados do IBOVESPA conforme especificado no EAP 1.0.
    """
    
    def __init__(self, caminho_dados=None):
        """
        Inicializa a classe com o caminho para os dados.
        
        Parâmetros:
        -----------
        caminho_dados : str
            Caminho para o arquivo de dados do IBOVESPA
        """
        self.caminho_dados = caminho_dados
        self.dados_brutos = None
        self.dados_limpos = None
        self.estatisticas_descritivas = None
        self.resultados_estacionariedade = {}
        
    def carregar_dados(self, caminho_dados=None):
        """
        1.1.1 - Carregar o dataset de 15 anos em um ambiente de análise (Pandas DataFrame).
        
        Parâmetros:
        -----------
        caminho_dados : str, opcional
            Caminho para o arquivo de dados. Se não fornecido, usa o caminho da inicialização.
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com os dados carregados
        """
        if caminho_dados:
            self.caminho_dados = caminho_dados
            
        if not self.caminho_dados:
            raise ValueError("Caminho para os dados deve ser fornecido")
            
        try:
            # Carrega os dados assumindo formato CSV com cabeçalho
            self.dados_brutos = pd.read_csv(self.caminho_dados)
            print(f"✓ Dados carregados com sucesso: {len(self.dados_brutos)} registros")
            print(f"✓ Colunas disponíveis: {list(self.dados_brutos.columns)}")
            print(f"✓ Período dos dados: {self.dados_brutos.iloc[0, 0]} até {self.dados_brutos.iloc[-1, 0]}")
            
            return self.dados_brutos
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            raise
    
    def converter_data_indice(self):
        """
        1.1.2 - Converter a coluna de data para o formato datetime e defini-la como índice.
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com índice de data convertido
        """
        if self.dados_brutos is None:
            raise ValueError("Dados devem ser carregados primeiro")
        
        # Identifica a coluna de data (primeira coluna ou coluna com 'date' no nome)
        colunas_data = [col for col in self.dados_brutos.columns if 'date' in col.lower()]
        if colunas_data:
            coluna_data = colunas_data[0]
        else:
            coluna_data = self.dados_brutos.columns[0]  # Assume primeira coluna
        
        try:
            # Converte para datetime
            self.dados_brutos[coluna_data] = pd.to_datetime(self.dados_brutos[coluna_data])
            
            # Define como índice
            self.dados_brutos.set_index(coluna_data, inplace=True)
            
            print(f"✓ Coluna de data '{coluna_data}' convertida para datetime e definida como índice")
            print(f"✓ Período: {self.dados_brutos.index.min()} até {self.dados_brutos.index.max()}")
            
            return self.dados_brutos
            
        except Exception as e:
            print(f"❌ Erro na conversão de data: {str(e)}")
            raise
    
    def verificar_integridade(self):
        """
        1.1.3 - Realizar verificação de integridade: checar duplicatas e 
        consistência dos dados (Máxima >= Fechamento >= Mínima).
        
        Retorna:
        --------
        dict
            Dicionário com resultados da verificação
        """
        if self.dados_brutos is None:
            raise ValueError("Dados devem ser carregados primeiro")
        
        resultados = {}
        
        # Verifica duplicatas no índice
        duplicatas_indice = self.dados_brutos.index.duplicated().sum()
        resultados['duplicatas_indice'] = duplicatas_indice
        
        # Verifica duplicatas em linhas completas
        duplicatas_linhas = self.dados_brutos.duplicated().sum()
        resultados['duplicatas_linhas'] = duplicatas_linhas
        
        # Verifica consistência de preços (assumindo colunas padrão)
        colunas_preco = ['High', 'Low', 'Close', 'Open']
        colunas_disponiveis = [col for col in colunas_preco if col in self.dados_brutos.columns]
        
        if len(colunas_disponiveis) >= 3:
            # Verifica se High >= Close >= Low
            if 'High' in colunas_disponiveis and 'Low' in colunas_disponiveis and 'Close' in colunas_disponiveis:
                inconsistencias_high = (self.dados_brutos['High'] < self.dados_brutos['Close']).sum()
                inconsistencias_low = (self.dados_brutos['Low'] > self.dados_brutos['Close']).sum()
                
                resultados['inconsistencias_high_close'] = inconsistencias_high
                resultados['inconsistencias_low_close'] = inconsistencias_low
                resultados['total_inconsistencias'] = inconsistencias_high + inconsistencias_low
        
        # Verifica valores nulos
        valores_nulos = self.dados_brutos.isnull().sum()
        resultados['valores_nulos'] = valores_nulos.to_dict()
        
        # Relatório
        print("=== VERIFICAÇÃO DE INTEGRIDADE DOS DADOS ===")
        print(f"✓ Duplicatas no índice: {duplicatas_indice}")
        print(f"✓ Duplicatas em linhas: {duplicatas_linhas}")
        
        if 'total_inconsistencias' in resultados:
            print(f"✓ Inconsistências de preço: {resultados['total_inconsistencias']}")
        
        print(f"✓ Valores nulos por coluna:")
        for coluna, nulos in resultados['valores_nulos'].items():
            if nulos > 0:
                print(f"   - {coluna}: {nulos}")
        
        return resultados
    
    def identificar_lacunas_temporais(self):
        """
        1.2.1 - Identificar lacunas na série temporal (dias sem pregão).
        
        Retorna:
        --------
        dict
            Informações sobre lacunas na série temporal
        """
        if self.dados_brutos is None:
            raise ValueError("Dados devem ser carregados primeiro")
        
        # Cria um índice completo de dias úteis
        inicio = self.dados_brutos.index.min()
        fim = self.dados_brutos.index.max()
        
        # Gera todas as datas no período
        todas_datas = pd.date_range(start=inicio, end=fim, freq='D')
        
        # Identifica datas ausentes
        datas_ausentes = todas_datas.difference(self.dados_brutos.index)
        
        # Classifica as lacunas
        fins_semana = datas_ausentes[datas_ausentes.weekday >= 5]  # Sábado=5, Domingo=6
        dias_uteis_ausentes = datas_ausentes[datas_ausentes.weekday < 5]
        
        resultados = {
            'total_datas_ausentes': len(datas_ausentes),
            'fins_semana_ausentes': len(fins_semana),
            'dias_uteis_ausentes': len(dias_uteis_ausentes),
            'datas_dias_uteis_ausentes': dias_uteis_ausentes.tolist()
        }
        
        print("=== ANÁLISE DE LACUNAS TEMPORAIS ===")
        print(f"✓ Total de datas ausentes: {resultados['total_datas_ausentes']}")
        print(f"✓ Fins de semana ausentes: {resultados['fins_semana_ausentes']}")
        print(f"✓ Dias úteis ausentes (feriados): {resultados['dias_uteis_ausentes']}")
        
        if len(dias_uteis_ausentes) > 0:
            print("✓ Primeiros 10 dias úteis ausentes:")
            for data in dias_uteis_ausentes[:10]:
                print(f"   - {data.strftime('%Y-%m-%d (%A)')}")
        
        return resultados
    
    def aplicar_forward_fill(self):
        """
        1.2.2 - Aplicar a estratégia de preenchimento para a frente (forward-fill / ffill) 
        para preencher os valores em feriados e fins de semana.
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com lacunas preenchidas
        """
        if self.dados_brutos is None:
            raise ValueError("Dados devem ser carregados primeiro")
        
        # Cria um índice completo de dias
        inicio = self.dados_brutos.index.min()
        fim = self.dados_brutos.index.max()
        indice_completo = pd.date_range(start=inicio, end=fim, freq='D')
        
        # Reindexiza com o índice completo e aplica forward fill
        dados_reindexados = self.dados_brutos.reindex(indice_completo)
        self.dados_limpos = dados_reindexados.fillna(method='ffill')
        
        # Estatísticas do preenchimento
        registros_originais = len(self.dados_brutos)
        registros_preenchidos = len(self.dados_limpos) - registros_originais
        
        print("=== APLICAÇÃO DE FORWARD FILL ===")
        print(f"✓ Registros originais: {registros_originais}")
        print(f"✓ Registros após preenchimento: {len(self.dados_limpos)}")
        print(f"✓ Registros preenchidos: {registros_preenchidos}")
        print(f"✓ Estratégia: Forward Fill (ffill) - valores mantidos do último pregão")
        
        return self.dados_limpos
    
    def documentar_decisao_outliers(self):
        """
        1.2.3 - Documentar a decisão de não remover outliers, 
        justificando-a pela natureza dos eventos de mercado.
        """
        documentacao = """
        === DECISÃO SOBRE TRATAMENTO DE OUTLIERS ===
        
        DECISÃO: NÃO REMOVER OUTLIERS
        
        JUSTIFICATIVA:
        1. Natureza dos Dados Financeiros:
           - Outliers em séries financeiras frequentemente representam eventos 
             legítimos de mercado (crises, instabilidade política, eventos macroeconômicos)
           
        2. Informação Valiosa:
           - Movimentos extremos contêm informações importantes sobre volatilidade
             e comportamento do mercado em momentos de estresse
           
        3. Robustez do Modelo:
           - Modelos baseados em árvores (XGBoost, Random Forest) são naturalmente
             robustos a outliers, não sendo necessária sua remoção
           
        4. Realismo:
           - Manter outliers garante que o modelo seja treinado com a realidade
             completa do mercado, incluindo eventos extremos
        
        ESTRATÉGIA ADOTADA:
        - Manter todos os dados originais
        - Utilizar modelos robustos a outliers na fase de modelagem
        - Monitorar impacto de eventos extremos na análise exploratória
        """
        
        print(documentacao)
        return documentacao

    def gerar_estatisticas_descritivas(self):
        """
        1.3.1 - Gerar e analisar estatísticas descritivas (.describe()).

        Retorna:
        --------
        pd.DataFrame
            Estatísticas descritivas dos dados
        """
        if self.dados_limpos is None:
            raise ValueError("Dados limpos devem estar disponíveis")

        self.estatisticas_descritivas = self.dados_limpos.describe()

        print("=== ESTATÍSTICAS DESCRITIVAS ===")
        print(self.estatisticas_descritivas)

        # Análise adicional
        print("\n=== ANÁLISE ADICIONAL ===")
        for coluna in self.dados_limpos.select_dtypes(include=[np.number]).columns:
            cv = self.dados_limpos[coluna].std() / self.dados_limpos[coluna].mean()
            print(f"✓ Coeficiente de Variação {coluna}: {cv:.4f}")

        return self.estatisticas_descritivas

    def plotar_preco_volume(self, salvar_grafico=False):
        """
        1.3.2 - Plotar o gráfico de linha do preço de fechamento (Close) e
        o gráfico de barras do volume (Volume) ao longo do tempo.

        Parâmetros:
        -----------
        salvar_grafico : bool
            Se True, salva o gráfico em arquivo
        """
        if self.dados_limpos is None:
            raise ValueError("Dados limpos devem estar disponíveis")

        # Identifica colunas de preço e volume
        col_close = None
        col_volume = None

        for col in self.dados_limpos.columns:
            if 'close' in col.lower():
                col_close = col
            elif 'volume' in col.lower():
                col_volume = col

        if col_close is None:
            print("❌ Coluna de preço de fechamento não encontrada")
            return

        # Cria subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Gráfico de preço
        ax1.plot(self.dados_limpos.index, self.dados_limpos[col_close],
                linewidth=1, color='blue', alpha=0.8)
        ax1.set_title('IBOVESPA - Preço de Fechamento ao Longo do Tempo', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Preço de Fechamento', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Gráfico de volume (se disponível)
        if col_volume is not None:
            # Converte volume para bilhões para melhor visualização
            volume_bilhoes = self.dados_limpos[col_volume] / 1e9

            # Usa fill_between ao invés de bar para séries temporais longas
            ax2.fill_between(self.dados_limpos.index, volume_bilhoes, alpha=0.7, color='orange')
            ax2.plot(self.dados_limpos.index, volume_bilhoes, linewidth=0.5, color='darkorange')

            ax2.set_title('IBOVESPA - Volume de Negociação ao Longo do Tempo', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Volume (Bilhões)', fontsize=12)
            ax2.set_xlabel('Data', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

            # Formatação do eixo Y
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}B'))
        else:
            ax2.text(0.5, 0.5, 'Dados de Volume não disponíveis',
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Volume não disponível', fontsize=14)

        plt.tight_layout()

        if salvar_grafico:
            plt.savefig('grafico_preco_volume.png', dpi=300, bbox_inches='tight')
            print("✓ Gráfico salvo como 'grafico_preco_volume.png'")

        plt.show()

        print("✓ Gráficos de preço e volume gerados com sucesso")

    def calcular_retornos_logaritmicos(self):
        """
        1.3.3 - Calcular os retornos logarítmicos diários.

        Retorna:
        --------
        pd.Series
            Série com retornos logarítmicos diários
        """
        if self.dados_limpos is None:
            raise ValueError("Dados limpos devem estar disponíveis")

        # Identifica coluna de fechamento
        col_close = None
        for col in self.dados_limpos.columns:
            if 'close' in col.lower():
                col_close = col
                break

        if col_close is None:
            raise ValueError("Coluna de preço de fechamento não encontrada")

        # Calcula retornos logarítmicos
        retornos = np.log(self.dados_limpos[col_close] / self.dados_limpos[col_close].shift(1))
        retornos = retornos.dropna()

        # Adiciona aos dados limpos
        self.dados_limpos['Retornos_Log'] = retornos

        print("=== RETORNOS LOGARÍTMICOS ===")
        print(f"✓ Retornos calculados: {len(retornos)} observações")
        print(f"✓ Estatísticas dos retornos:")
        print(f"   - Média: {retornos.mean():.6f}")
        print(f"   - Desvio Padrão: {retornos.std():.6f}")
        print(f"   - Mínimo: {retornos.min():.6f}")
        print(f"   - Máximo: {retornos.max():.6f}")
        print(f"   - Assimetria: {retornos.skew():.6f}")
        print(f"   - Curtose: {retornos.kurtosis():.6f}")

        return retornos

    def plotar_histograma_retornos(self, salvar_grafico=False):
        """
        1.3.4 - Plotar o histograma dos retornos para analisar a distribuição
        (verificar curtose e "caudas gordas").

        Parâmetros:
        -----------
        salvar_grafico : bool
            Se True, salva o gráfico em arquivo
        """
        if 'Retornos_Log' not in self.dados_limpos.columns:
            print("❌ Retornos logarítmicos devem ser calculados primeiro")
            return

        retornos = self.dados_limpos['Retornos_Log'].dropna()

        # Cria o gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histograma
        ax1.hist(retornos, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribuição dos Retornos Logarítmicos Diários', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Retornos Logarítmicos', fontsize=12)
        ax1.set_ylabel('Densidade', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Adiciona curva normal para comparação
        mu, sigma = retornos.mean(), retornos.std()
        x = np.linspace(retornos.min(), retornos.max(), 100)
        normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax1.plot(x, normal_curve, 'r-', linewidth=2, label='Distribuição Normal')
        ax1.legend()

        # Q-Q plot
        from scipy import stats
        stats.probplot(retornos, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Retornos vs Distribuição Normal', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if salvar_grafico:
            plt.savefig('histograma_retornos.png', dpi=300, bbox_inches='tight')
            print("✓ Gráfico salvo como 'histograma_retornos.png'")

        plt.show()

        # Análise estatística
        print("=== ANÁLISE DA DISTRIBUIÇÃO DOS RETORNOS ===")
        print(f"✓ Curtose: {retornos.kurtosis():.4f} (Normal = 0)")
        if retornos.kurtosis() > 0:
            print("   → Distribuição leptocúrtica (caudas gordas)")
        else:
            print("   → Distribuição platicúrtica (caudas finas)")

        print(f"✓ Assimetria: {retornos.skew():.4f} (Normal = 0)")
        if abs(retornos.skew()) > 0.5:
            print("   → Distribuição assimétrica")
        else:
            print("   → Distribuição aproximadamente simétrica")

        # Teste de normalidade
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(retornos)
        print(f"✓ Teste Jarque-Bera: estatística={jb_stat:.4f}, p-valor={jb_pvalue:.6f}")
        if jb_pvalue < 0.05:
            print("   → Rejeita hipótese de normalidade (p < 0.05)")
        else:
            print("   → Não rejeita hipótese de normalidade (p >= 0.05)")

    def plotar_serie_retornos(self, salvar_grafico=False):
        """
        1.3.5 - Plotar o gráfico de linha dos retornos para identificar visualmente
        o agrupamento de volatilidade (volatility clustering).

        Parâmetros:
        -----------
        salvar_grafico : bool
            Se True, salva o gráfico em arquivo
        """
        if 'Retornos_Log' not in self.dados_limpos.columns:
            print("❌ Retornos logarítmicos devem ser calculados primeiro")
            return

        retornos = self.dados_limpos['Retornos_Log'].dropna()

        # Cria o gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Série temporal dos retornos
        ax1.plot(retornos.index, retornos, linewidth=0.8, color='darkblue', alpha=0.8)
        ax1.set_title('IBOVESPA - Retornos Logarítmicos Diários ao Longo do Tempo',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Retornos Logarítmicos', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # Volatilidade móvel (desvio padrão móvel de 30 dias)
        volatilidade_movel = retornos.rolling(window=30).std()
        ax2.plot(volatilidade_movel.index, volatilidade_movel,
                linewidth=1.5, color='orange', alpha=0.8)
        ax2.set_title('Volatilidade Móvel (30 dias) - Evidência de Agrupamento de Volatilidade',
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatilidade (Desvio Padrão)', fontsize=12)
        ax2.set_xlabel('Data', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if salvar_grafico:
            plt.savefig('serie_retornos_volatilidade.png', dpi=300, bbox_inches='tight')
            print("✓ Gráfico salvo como 'serie_retornos_volatilidade.png'")

        plt.show()

        print("=== ANÁLISE DE AGRUPAMENTO DE VOLATILIDADE ===")
        print("✓ Gráfico de retornos e volatilidade móvel gerado")
        print("✓ Observe períodos de alta volatilidade seguidos por alta volatilidade")
        print("✓ Este padrão é conhecido como 'volatility clustering'")
        print("✓ Justifica o uso de features baseadas em volatilidade na modelagem")

    def teste_estacionariedade_precos(self):
        """
        1.4.1 - Aplicar os testes ADF (Augmented Dickey-Fuller) e KPSS
        (Kwiatkowski-Phillips-Schmidt-Shin) na série de preços de fechamento.

        Retorna:
        --------
        dict
            Resultados dos testes de estacionariedade para preços
        """
        if self.dados_limpos is None:
            raise ValueError("Dados limpos devem estar disponíveis")

        # Identifica coluna de fechamento
        col_close = None
        for col in self.dados_limpos.columns:
            if 'close' in col.lower():
                col_close = col
                break

        if col_close is None:
            raise ValueError("Coluna de preço de fechamento não encontrada")

        precos = self.dados_limpos[col_close].dropna()

        # Teste ADF
        try:
            adf_result = adfuller(precos, autolag='AIC')
            adf_estatistica = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_valores_criticos = adf_result[4]
        except Exception as e:
            print(f"❌ Erro no teste ADF: {str(e)}")
            adf_estatistica = adf_pvalue = adf_valores_criticos = None

        # Teste KPSS
        try:
            kpss_result = kpss(precos, regression='c', nlags='auto')
            kpss_estatistica = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_valores_criticos = kpss_result[3]
        except Exception as e:
            print(f"❌ Erro no teste KPSS: {str(e)}")
            kpss_estatistica = kpss_pvalue = kpss_valores_criticos = None

        # Armazena resultados
        self.resultados_estacionariedade['precos'] = {
            'adf_estatistica': adf_estatistica,
            'adf_pvalue': adf_pvalue,
            'adf_valores_criticos': adf_valores_criticos,
            'kpss_estatistica': kpss_estatistica,
            'kpss_pvalue': kpss_pvalue,
            'kpss_valores_criticos': kpss_valores_criticos
        }

        # Relatório
        print("=== TESTE DE ESTACIONARIEDADE - PREÇOS ===")

        if adf_estatistica is not None:
            print(f"✓ Teste ADF (Augmented Dickey-Fuller):")
            print(f"   - Estatística: {adf_estatistica:.6f}")
            print(f"   - P-valor: {adf_pvalue:.6f}")
            print(f"   - Valores críticos: {adf_valores_criticos}")

            if adf_pvalue <= 0.05:
                print("   → Rejeita H0: Série é ESTACIONÁRIA (p <= 0.05)")
            else:
                print("   → Não rejeita H0: Série é NÃO ESTACIONÁRIA (p > 0.05)")

        if kpss_estatistica is not None:
            print(f"\n✓ Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin):")
            print(f"   - Estatística: {kpss_estatistica:.6f}")
            print(f"   - P-valor: {kpss_pvalue:.6f}")
            print(f"   - Valores críticos: {kpss_valores_criticos}")

            if kpss_pvalue <= 0.05:
                print("   → Rejeita H0: Série é NÃO ESTACIONÁRIA (p <= 0.05)")
            else:
                print("   → Não rejeita H0: Série é ESTACIONÁRIA (p > 0.05)")

        return self.resultados_estacionariedade['precos']

    def teste_estacionariedade_retornos(self):
        """
        1.4.2 - Aplicar os testes ADF e KPSS na série de retornos diários.

        Retorna:
        --------
        dict
            Resultados dos testes de estacionariedade para retornos
        """
        if 'Retornos_Log' not in self.dados_limpos.columns:
            print("❌ Retornos logarítmicos devem ser calculados primeiro")
            return None

        retornos = self.dados_limpos['Retornos_Log'].dropna()

        # Teste ADF
        try:
            adf_result = adfuller(retornos, autolag='AIC')
            adf_estatistica = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_valores_criticos = adf_result[4]
        except Exception as e:
            print(f"❌ Erro no teste ADF: {str(e)}")
            adf_estatistica = adf_pvalue = adf_valores_criticos = None

        # Teste KPSS
        try:
            kpss_result = kpss(retornos, regression='c', nlags='auto')
            kpss_estatistica = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_valores_criticos = kpss_result[3]
        except Exception as e:
            print(f"❌ Erro no teste KPSS: {str(e)}")
            kpss_estatistica = kpss_pvalue = kpss_valores_criticos = None

        # Armazena resultados
        self.resultados_estacionariedade['retornos'] = {
            'adf_estatistica': adf_estatistica,
            'adf_pvalue': adf_pvalue,
            'adf_valores_criticos': adf_valores_criticos,
            'kpss_estatistica': kpss_estatistica,
            'kpss_pvalue': kpss_pvalue,
            'kpss_valores_criticos': kpss_valores_criticos
        }

        # Relatório
        print("=== TESTE DE ESTACIONARIEDADE - RETORNOS ===")

        if adf_estatistica is not None:
            print(f"✓ Teste ADF (Augmented Dickey-Fuller):")
            print(f"   - Estatística: {adf_estatistica:.6f}")
            print(f"   - P-valor: {adf_pvalue:.6f}")
            print(f"   - Valores críticos: {adf_valores_criticos}")

            if adf_pvalue <= 0.05:
                print("   → Rejeita H0: Série é ESTACIONÁRIA (p <= 0.05)")
            else:
                print("   → Não rejeita H0: Série é NÃO ESTACIONÁRIA (p > 0.05)")

        if kpss_estatistica is not None:
            print(f"\n✓ Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin):")
            print(f"   - Estatística: {kpss_estatistica:.6f}")
            print(f"   - P-valor: {kpss_pvalue:.6f}")
            print(f"   - Valores críticos: {kpss_valores_criticos}")

            if kpss_pvalue <= 0.05:
                print("   → Rejeita H0: Série é NÃO ESTACIONÁRIA (p <= 0.05)")
            else:
                print("   → Não rejeita H0: Série é ESTACIONÁRIA (p > 0.05)")

        return self.resultados_estacionariedade['retornos']

    def documentar_resultados_estacionariedade(self):
        """
        1.4.3 - Documentar os resultados, confirmando a não estacionariedade
        dos preços e a estacionariedade dos retornos.

        Retorna:
        --------
        str
            Documentação dos resultados dos testes
        """
        if not self.resultados_estacionariedade:
            print("❌ Testes de estacionariedade devem ser executados primeiro")
            return None

        documentacao = """
        === DOCUMENTAÇÃO DOS RESULTADOS DE ESTACIONARIEDADE ===

        RESUMO DOS ACHADOS:

        1. SÉRIE DE PREÇOS:
           - Resultado esperado: NÃO ESTACIONÁRIA
           - Justificativa: Preços de ativos financeiros tipicamente apresentam
             tendências de longo prazo e não possuem média constante
           - Implicação: Não deve ser usada diretamente na modelagem

        2. SÉRIE DE RETORNOS:
           - Resultado esperado: ESTACIONÁRIA
           - Justificativa: A diferenciação (cálculo de retornos) remove
             tendências e torna a série estacionária
           - Implicação: Pode ser usada na modelagem e criação de features

        CONCLUSÕES PARA MODELAGEM:

        ✓ Usar retornos (não preços) como base para features
        ✓ Indicadores técnicos devem ser baseados em transformações estacionárias
        ✓ Evitar uso direto de níveis de preço como features
        ✓ Confirma a necessidade de trabalhar com diferenças/retornos

        PRÓXIMOS PASSOS:
        - Fase 2: Criar variável alvo baseada em direção (não magnitude)
        - Fase 2: Desenvolver features baseadas em retornos e indicadores técnicos
        - Fase 2: Garantir que todas as features sejam estacionárias
        """

        print(documentacao)

        # Adiciona resultados específicos se disponíveis
        if 'precos' in self.resultados_estacionariedade:
            precos_res = self.resultados_estacionariedade['precos']
            if precos_res['adf_pvalue'] is not None:
                print(f"\nRESULTADOS ESPECÍFICOS - PREÇOS:")
                print(f"ADF p-valor: {precos_res['adf_pvalue']:.6f}")
                if precos_res['kpss_pvalue'] is not None:
                    print(f"KPSS p-valor: {precos_res['kpss_pvalue']:.6f}")

        if 'retornos' in self.resultados_estacionariedade:
            retornos_res = self.resultados_estacionariedade['retornos']
            if retornos_res['adf_pvalue'] is not None:
                print(f"\nRESULTADOS ESPECÍFICOS - RETORNOS:")
                print(f"ADF p-valor: {retornos_res['adf_pvalue']:.6f}")
                if retornos_res['kpss_pvalue'] is not None:
                    print(f"KPSS p-valor: {retornos_res['kpss_pvalue']:.6f}")

        return documentacao

    def executar_fase_1_completa(self, caminho_dados, salvar_graficos=True):
        """
        Executa toda a Fase 1 do EAP de forma sequencial.

        Parâmetros:
        -----------
        caminho_dados : str
            Caminho para o arquivo de dados do IBOVESPA
        salvar_graficos : bool
            Se True, salva os gráficos gerados

        Retorna:
        --------
        dict
            Resumo completo da execução da Fase 1
        """
        print("=" * 80)
        print("INICIANDO FASE 1: INTEGRIDADE DOS DADOS E ANÁLISE EXPLORATÓRIA")
        print("=" * 80)

        resultados_fase1 = {}

        try:
            # 1.1 - Aquisição e Validação Inicial
            print("\n" + "=" * 50)
            print("1.1 - AQUISIÇÃO E VALIDAÇÃO INICIAL")
            print("=" * 50)

            # 1.1.1 - Carregar dados
            self.carregar_dados(caminho_dados)
            resultados_fase1['dados_carregados'] = True

            # 1.1.2 - Converter data para índice
            self.converter_data_indice()
            resultados_fase1['data_convertida'] = True

            # 1.1.3 - Verificar integridade
            integridade = self.verificar_integridade()
            resultados_fase1['verificacao_integridade'] = integridade

            # 1.2 - Limpeza e Tratamento de Dados
            print("\n" + "=" * 50)
            print("1.2 - LIMPEZA E TRATAMENTO DE DADOS")
            print("=" * 50)

            # 1.2.1 - Identificar lacunas
            lacunas = self.identificar_lacunas_temporais()
            resultados_fase1['lacunas_temporais'] = lacunas

            # 1.2.2 - Aplicar forward fill
            self.aplicar_forward_fill()
            resultados_fase1['forward_fill_aplicado'] = True

            # 1.2.3 - Documentar decisão sobre outliers
            doc_outliers = self.documentar_decisao_outliers()
            resultados_fase1['documentacao_outliers'] = doc_outliers

            # 1.3 - Análise Exploratória de Dados (EDA)
            print("\n" + "=" * 50)
            print("1.3 - ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
            print("=" * 50)

            # 1.3.1 - Estatísticas descritivas
            stats = self.gerar_estatisticas_descritivas()
            resultados_fase1['estatisticas_descritivas'] = stats

            # 1.3.2 - Gráficos de preço e volume
            self.plotar_preco_volume(salvar_grafico=salvar_graficos)
            resultados_fase1['graficos_preco_volume'] = True

            # 1.3.3 - Calcular retornos logarítmicos
            retornos = self.calcular_retornos_logaritmicos()
            resultados_fase1['retornos_calculados'] = True

            # 1.3.4 - Histograma dos retornos
            self.plotar_histograma_retornos(salvar_grafico=salvar_graficos)
            resultados_fase1['histograma_retornos'] = True

            # 1.3.5 - Série temporal dos retornos
            self.plotar_serie_retornos(salvar_grafico=salvar_graficos)
            resultados_fase1['serie_retornos'] = True

            # 1.4 - Teste de Estacionariedade
            print("\n" + "=" * 50)
            print("1.4 - TESTE DE ESTACIONARIEDADE")
            print("=" * 50)

            # 1.4.1 - Teste nos preços
            teste_precos = self.teste_estacionariedade_precos()
            resultados_fase1['teste_estacionariedade_precos'] = teste_precos

            # 1.4.2 - Teste nos retornos
            teste_retornos = self.teste_estacionariedade_retornos()
            resultados_fase1['teste_estacionariedade_retornos'] = teste_retornos

            # 1.4.3 - Documentar resultados
            doc_estacionariedade = self.documentar_resultados_estacionariedade()
            resultados_fase1['documentacao_estacionariedade'] = doc_estacionariedade

            # Resumo final
            print("\n" + "=" * 80)
            print("FASE 1 CONCLUÍDA COM SUCESSO!")
            print("=" * 80)
            print("✓ Todos os itens do EAP 1.0 foram executados")
            print("✓ Dados validados e limpos")
            print("✓ Análise exploratória completa")
            print("✓ Testes de estacionariedade realizados")
            print("✓ Documentação gerada")
            if salvar_graficos:
                print("✓ Gráficos salvos na pasta do projeto")

            resultados_fase1['fase_1_completa'] = True
            resultados_fase1['status'] = 'SUCESSO'

        except Exception as e:
            print(f"\n❌ ERRO NA EXECUÇÃO DA FASE 1: {str(e)}")
            resultados_fase1['status'] = 'ERRO'
            resultados_fase1['erro'] = str(e)
            raise

        return resultados_fase1


def main():
    """
    Função principal para demonstração da Fase 1.
    """
    print("DEMONSTRAÇÃO DA FASE 1 - INTEGRIDADE DOS DADOS E EDA")
    print("=" * 60)

    # Exemplo de uso (ajuste o caminho conforme necessário)
    caminho_exemplo = "dados_ibovespa.csv"  # Substitua pelo caminho real

    # Cria instância da classe
    eda = IntegridadeDadosEDA()

    print(f"Para executar a Fase 1 completa, use:")
    print(f"eda = IntegridadeDadosEDA()")
    print(f"resultados = eda.executar_fase_1_completa('{caminho_exemplo}')")
    print("\nOu execute cada etapa individualmente conforme documentado no EAP.")


if __name__ == "__main__":
    main()
