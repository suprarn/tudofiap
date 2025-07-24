#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 1 COMPLETA - INTEGRIDADE DOS DADOS E ANÁLISE EXPLORATÓRIA
==============================================================

Arquivo único organizado em blocos para facilitar conversão para Jupyter Notebook.
Implementa todos os itens do EAP 1.0 de forma sequencial e modular.

Autor: Projeto Tech Challenge 2
Data: 2025-01-24
Referência: EAP.md e Steering.md
"""

# =============================================================================
# BLOCO 1: IMPORTAÇÕES E CONFIGURAÇÕES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos em português
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("✅ Bibliotecas importadas e configurações definidas")

# =============================================================================
# BLOCO 2: CRIAÇÃO DE DADOS DE EXEMPLO (OPCIONAL)
# =============================================================================

def criar_dados_exemplo_ibovespa(n_anos=15):
    """
    Cria um dataset de exemplo do IBOVESPA para demonstração.
    Use este bloco apenas se não tiver dados reais.
    """
    print("=== CRIANDO DADOS DE EXEMPLO DO IBOVESPA ===")
    
    # Período de dados
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=n_anos * 365)
    
    # Gera datas de pregão (apenas dias úteis)
    datas = pd.bdate_range(start=data_inicio, end=data_fim, freq='B')
    
    # Parâmetros para simulação realística
    np.random.seed(42)  # Para reprodutibilidade
    n_dias = len(datas)
    
    # Simula preços com random walk + tendência
    preco_inicial = 100000  # IBOVESPA em torno de 100k pontos
    retornos_diarios = np.random.normal(0.0005, 0.015, n_dias)  # ~0.05% média, 1.5% volatilidade
    
    # Adiciona alguns eventos extremos (crises)
    eventos_extremos = np.random.choice(n_dias, size=int(n_dias * 0.02), replace=False)
    retornos_diarios[eventos_extremos] = np.random.normal(0, 0.05, len(eventos_extremos))
    
    # Calcula preços
    precos = [preco_inicial]
    for retorno in retornos_diarios[1:]:
        novo_preco = precos[-1] * (1 + retorno)
        precos.append(novo_preco)
    
    precos = np.array(precos)
    
    # Simula OHLC baseado no Close
    high = precos * (1 + np.abs(np.random.normal(0, 0.01, n_dias)))
    low = precos * (1 - np.abs(np.random.normal(0, 0.01, n_dias)))
    open_prices = np.roll(precos, 1)  # Open é o close anterior
    open_prices[0] = precos[0]
    
    # Simula volume com variação realística
    volume_base = 2000000000  # 2 bilhões (volume típico)
    volume = np.random.lognormal(np.log(volume_base), 0.5, n_dias)  # Maior variação
    
    # Cria DataFrame
    dados_ibovespa = pd.DataFrame({
        'Date': datas,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': precos,
        'Volume': volume.astype(int)
    })
    
    # Garante consistência OHLC
    dados_ibovespa['High'] = dados_ibovespa[['Open', 'High', 'Low', 'Close']].max(axis=1)
    dados_ibovespa['Low'] = dados_ibovespa[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    print(f"✓ Dados simulados criados: {len(dados_ibovespa)} registros")
    print(f"✓ Período: {dados_ibovespa['Date'].min()} até {dados_ibovespa['Date'].max()}")
    print(f"✓ Preço inicial: {dados_ibovespa['Close'].iloc[0]:,.0f}")
    print(f"✓ Preço final: {dados_ibovespa['Close'].iloc[-1]:,.0f}")
    
    # Salva arquivo
    dados_ibovespa.to_csv('dados_ibovespa_exemplo.csv', index=False)
    print("✓ Arquivo salvo: dados_ibovespa_exemplo.csv")
    
    return dados_ibovespa

# Descomente a linha abaixo se quiser criar dados de exemplo
# dados_exemplo = criar_dados_exemplo_ibovespa()

# =============================================================================
# BLOCO 3: CARREGAMENTO E VALIDAÇÃO INICIAL DOS DADOS (EAP 1.1)
# =============================================================================

def carregar_e_validar_dados(caminho_arquivo):
    """
    EAP 1.1.1, 1.1.2, 1.1.3 - Carregamento, conversão de data e verificação de integridade
    """
    print("=" * 60)
    print("EAP 1.1 - AQUISIÇÃO E VALIDAÇÃO INICIAL")
    print("=" * 60)
    
    # 1.1.1 - Carregar o dataset
    print("1.1.1 - Carregando dados...")
    dados = pd.read_csv(caminho_arquivo)
    print(f"✓ Dados carregados: {len(dados)} registros")
    print(f"✓ Colunas: {list(dados.columns)}")
    
    # 1.1.2 - Converter data para datetime e definir como índice
    print("\n1.1.2 - Convertendo data para índice...")
    
    # Identifica coluna de data
    colunas_data = [col for col in dados.columns if 'date' in col.lower()]
    if colunas_data:
        coluna_data = colunas_data[0]
    else:
        coluna_data = dados.columns[0]  # Assume primeira coluna
    
    dados[coluna_data] = pd.to_datetime(dados[coluna_data])
    dados.set_index(coluna_data, inplace=True)
    print(f"✓ Data convertida e definida como índice")
    print(f"✓ Período: {dados.index.min()} até {dados.index.max()}")
    
    # 1.1.3 - Verificação de integridade
    print("\n1.1.3 - Verificando integridade dos dados...")
    
    # Duplicatas
    duplicatas_indice = dados.index.duplicated().sum()
    duplicatas_linhas = dados.duplicated().sum()
    
    # Consistência OHLC
    inconsistencias = 0
    if all(col in dados.columns for col in ['High', 'Low', 'Close']):
        inconsistencias_high = (dados['High'] < dados['Close']).sum()
        inconsistencias_low = (dados['Low'] > dados['Close']).sum()
        inconsistencias = inconsistencias_high + inconsistencias_low
    
    # Valores nulos
    valores_nulos = dados.isnull().sum()
    
    print(f"✓ Duplicatas no índice: {duplicatas_indice}")
    print(f"✓ Duplicatas em linhas: {duplicatas_linhas}")
    print(f"✓ Inconsistências OHLC: {inconsistencias}")
    print(f"✓ Valores nulos por coluna:")
    for coluna, nulos in valores_nulos.items():
        if nulos > 0:
            print(f"   - {coluna}: {nulos}")
    
    return dados

# =============================================================================
# BLOCO 4: LIMPEZA E TRATAMENTO DE DADOS (EAP 1.2)
# =============================================================================

def limpar_e_tratar_dados(dados):
    """
    EAP 1.2.1, 1.2.2, 1.2.3 - Identificação de lacunas, forward fill e documentação de outliers
    """
    print("\n" + "=" * 60)
    print("EAP 1.2 - LIMPEZA E TRATAMENTO DE DADOS")
    print("=" * 60)
    
    # 1.2.1 - Identificar lacunas na série temporal
    print("1.2.1 - Identificando lacunas temporais...")
    
    inicio = dados.index.min()
    fim = dados.index.max()
    todas_datas = pd.date_range(start=inicio, end=fim, freq='D')
    datas_ausentes = todas_datas.difference(dados.index)
    
    # Classifica lacunas
    fins_semana = datas_ausentes[datas_ausentes.weekday >= 5]
    dias_uteis_ausentes = datas_ausentes[datas_ausentes.weekday < 5]
    
    print(f"✓ Total de datas ausentes: {len(datas_ausentes)}")
    print(f"✓ Fins de semana ausentes: {len(fins_semana)}")
    print(f"✓ Dias úteis ausentes (feriados): {len(dias_uteis_ausentes)}")
    
    # 1.2.2 - Aplicar forward fill
    print("\n1.2.2 - Aplicando forward fill...")
    
    registros_originais = len(dados)
    indice_completo = pd.date_range(start=inicio, end=fim, freq='D')
    dados_reindexados = dados.reindex(indice_completo)
    dados_limpos = dados_reindexados.fillna(method='ffill')
    
    print(f"✓ Registros originais: {registros_originais}")
    print(f"✓ Registros após preenchimento: {len(dados_limpos)}")
    print(f"✓ Registros preenchidos: {len(dados_limpos) - registros_originais}")
    
    # 1.2.3 - Documentar decisão sobre outliers
    print("\n1.2.3 - Documentando decisão sobre outliers...")
    print("""
    DECISÃO: NÃO REMOVER OUTLIERS
    
    JUSTIFICATIVA:
    • Outliers em dados financeiros representam eventos legítimos de mercado
    • Movimentos extremos contêm informação valiosa sobre volatilidade
    • Modelos baseados em árvores (XGBoost) são robustos a outliers
    • Manter outliers garante realismo na modelagem
    """)
    
    return dados_limpos

# =============================================================================
# BLOCO 5: ESTATÍSTICAS DESCRITIVAS (EAP 1.3.1)
# =============================================================================

def gerar_estatisticas_descritivas(dados):
    """
    EAP 1.3.1 - Gerar e analisar estatísticas descritivas
    """
    print("\n" + "=" * 60)
    print("EAP 1.3.1 - ESTATÍSTICAS DESCRITIVAS")
    print("=" * 60)
    
    stats_desc = dados.describe()
    print("Estatísticas Descritivas:")
    print(stats_desc)
    
    print("\nCoeficientes de Variação:")
    for coluna in dados.select_dtypes(include=[np.number]).columns:
        cv = dados[coluna].std() / dados[coluna].mean()
        print(f"✓ {coluna}: {cv:.4f}")
    
    return stats_desc

# =============================================================================
# BLOCO 6: GRÁFICOS DE PREÇO E VOLUME (EAP 1.3.2)
# =============================================================================

def plotar_preco_e_volume(dados, salvar_grafico=True):
    """
    EAP 1.3.2 - Plotar gráfico de preço de fechamento e volume ao longo do tempo
    """
    print("\n" + "=" * 60)
    print("EAP 1.3.2 - GRÁFICOS DE PREÇO E VOLUME")
    print("=" * 60)

    # Identifica colunas
    col_close = None
    col_volume = None

    for col in dados.columns:
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
    ax1.plot(dados.index, dados[col_close], linewidth=1, color='blue', alpha=0.8)
    ax1.set_title('IBOVESPA - Preço de Fechamento ao Longo do Tempo', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Preço de Fechamento', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Formatação do eixo Y para preços
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Gráfico de volume (corrigido)
    if col_volume is not None:
        # Converte volume para bilhões para melhor visualização
        volume_bilhoes = dados[col_volume] / 1e9

        # Usa plot ao invés de bar para séries temporais longas
        ax2.fill_between(dados.index, volume_bilhoes, alpha=0.7, color='orange')
        ax2.plot(dados.index, volume_bilhoes, linewidth=0.5, color='darkorange')

        ax2.set_title('IBOVESPA - Volume de Negociação ao Longo do Tempo', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volume (Bilhões)', fontsize=12)
        ax2.set_xlabel('Data', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Formatação do eixo Y para volume
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}B'))

        # Estatísticas do volume
        print(f"✓ Volume médio: {dados[col_volume].mean()/1e9:.2f} bilhões")
        print(f"✓ Volume mínimo: {dados[col_volume].min()/1e9:.2f} bilhões")
        print(f"✓ Volume máximo: {dados[col_volume].max()/1e9:.2f} bilhões")
    else:
        ax2.text(0.5, 0.5, 'Dados de Volume não disponíveis',
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Volume não disponível', fontsize=14)

    plt.tight_layout()

    if salvar_grafico:
        plt.savefig('grafico_preco_volume.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico salvo como 'grafico_preco_volume.png'")

    plt.show()
    print("✓ Gráficos de preço e volume gerados")

# =============================================================================
# BLOCO 7: CÁLCULO DE RETORNOS LOGARÍTMICOS (EAP 1.3.3)
# =============================================================================

def calcular_retornos_logaritmicos(dados):
    """
    EAP 1.3.3 - Calcular retornos logarítmicos diários
    """
    print("\n" + "=" * 60)
    print("EAP 1.3.3 - RETORNOS LOGARÍTMICOS")
    print("=" * 60)

    # Identifica coluna de fechamento
    col_close = None
    for col in dados.columns:
        if 'close' in col.lower():
            col_close = col
            break

    if col_close is None:
        raise ValueError("Coluna de preço de fechamento não encontrada")

    # Calcula retornos logarítmicos
    retornos = np.log(dados[col_close] / dados[col_close].shift(1))
    retornos = retornos.dropna()

    # Adiciona aos dados
    dados['Retornos_Log'] = np.log(dados[col_close] / dados[col_close].shift(1))

    print(f"✓ Retornos calculados: {len(retornos)} observações")
    print(f"✓ Estatísticas dos retornos:")
    print(f"   - Média: {retornos.mean():.6f}")
    print(f"   - Desvio Padrão: {retornos.std():.6f}")
    print(f"   - Mínimo: {retornos.min():.6f}")
    print(f"   - Máximo: {retornos.max():.6f}")
    print(f"   - Assimetria: {retornos.skew():.6f}")
    print(f"   - Curtose: {retornos.kurtosis():.6f}")

    return dados, retornos

# =============================================================================
# BLOCO 8: ANÁLISE DA DISTRIBUIÇÃO DOS RETORNOS (EAP 1.3.4)
# =============================================================================

def analisar_distribuicao_retornos(retornos, salvar_grafico=True):
    """
    EAP 1.3.4 - Plotar histograma dos retornos e analisar distribuição
    """
    print("\n" + "=" * 60)
    print("EAP 1.3.4 - ANÁLISE DA DISTRIBUIÇÃO DOS RETORNOS")
    print("=" * 60)

    # Cria gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histograma
    ax1.hist(retornos, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribuição dos Retornos Logarítmicos Diários', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Retornos Logarítmicos', fontsize=12)
    ax1.set_ylabel('Densidade', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Curva normal para comparação
    mu, sigma = retornos.mean(), retornos.std()
    x = np.linspace(retornos.min(), retornos.max(), 100)
    normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax1.plot(x, normal_curve, 'r-', linewidth=2, label='Distribuição Normal')
    ax1.legend()

    # Q-Q plot
    stats.probplot(retornos, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Retornos vs Distribuição Normal', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if salvar_grafico:
        plt.savefig('histograma_retornos.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico salvo como 'histograma_retornos.png'")

    plt.show()

    # Análise estatística
    print("Análise da Distribuição:")
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

# =============================================================================
# BLOCO 9: ANÁLISE DE VOLATILITY CLUSTERING (EAP 1.3.5)
# =============================================================================

def analisar_volatility_clustering(retornos, salvar_grafico=True):
    """
    EAP 1.3.5 - Plotar série de retornos para identificar volatility clustering
    """
    print("\n" + "=" * 60)
    print("EAP 1.3.5 - ANÁLISE DE VOLATILITY CLUSTERING")
    print("=" * 60)

    # Cria gráficos
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

    print("✓ Análise de volatility clustering concluída")
    print("✓ Observe períodos de alta volatilidade seguidos por alta volatilidade")
    print("✓ Este padrão justifica o uso de features baseadas em volatilidade")

# =============================================================================
# BLOCO 10: TESTES DE ESTACIONARIEDADE - PREÇOS (EAP 1.4.1)
# =============================================================================

def testar_estacionariedade_precos(dados):
    """
    EAP 1.4.1 - Aplicar testes ADF e KPSS na série de preços de fechamento
    """
    print("\n" + "=" * 60)
    print("EAP 1.4.1 - TESTE DE ESTACIONARIEDADE - PREÇOS")
    print("=" * 60)

    # Identifica coluna de fechamento
    col_close = None
    for col in dados.columns:
        if 'close' in col.lower():
            col_close = col
            break

    if col_close is None:
        raise ValueError("Coluna de preço de fechamento não encontrada")

    precos = dados[col_close].dropna()

    # Teste ADF
    print("Teste ADF (Augmented Dickey-Fuller):")
    try:
        adf_result = adfuller(precos, autolag='AIC')
        adf_estatistica = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_valores_criticos = adf_result[4]

        print(f"✓ Estatística: {adf_estatistica:.6f}")
        print(f"✓ P-valor: {adf_pvalue:.6f}")
        print(f"✓ Valores críticos: {adf_valores_criticos}")

        if adf_pvalue <= 0.05:
            print("   → Rejeita H0: Série é ESTACIONÁRIA (p <= 0.05)")
        else:
            print("   → Não rejeita H0: Série é NÃO ESTACIONÁRIA (p > 0.05)")
    except Exception as e:
        print(f"❌ Erro no teste ADF: {str(e)}")
        adf_estatistica = adf_pvalue = adf_valores_criticos = None

    # Teste KPSS
    print("\nTeste KPSS (Kwiatkowski-Phillips-Schmidt-Shin):")
    try:
        kpss_result = kpss(precos, regression='c', nlags='auto')
        kpss_estatistica = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        kpss_valores_criticos = kpss_result[3]

        print(f"✓ Estatística: {kpss_estatistica:.6f}")
        print(f"✓ P-valor: {kpss_pvalue:.6f}")
        print(f"✓ Valores críticos: {kpss_valores_criticos}")

        if kpss_pvalue <= 0.05:
            print("   → Rejeita H0: Série é NÃO ESTACIONÁRIA (p <= 0.05)")
        else:
            print("   → Não rejeita H0: Série é ESTACIONÁRIA (p > 0.05)")
    except Exception as e:
        print(f"❌ Erro no teste KPSS: {str(e)}")
        kpss_estatistica = kpss_pvalue = kpss_valores_criticos = None

    return {
        'adf_estatistica': adf_estatistica,
        'adf_pvalue': adf_pvalue,
        'kpss_estatistica': kpss_estatistica,
        'kpss_pvalue': kpss_pvalue
    }

# =============================================================================
# BLOCO 11: TESTES DE ESTACIONARIEDADE - RETORNOS (EAP 1.4.2)
# =============================================================================

def testar_estacionariedade_retornos(dados):
    """
    EAP 1.4.2 - Aplicar testes ADF e KPSS na série de retornos diários
    """
    print("\n" + "=" * 60)
    print("EAP 1.4.2 - TESTE DE ESTACIONARIEDADE - RETORNOS")
    print("=" * 60)

    if 'Retornos_Log' not in dados.columns:
        print("❌ Retornos logarítmicos devem ser calculados primeiro")
        return None

    retornos = dados['Retornos_Log'].dropna()

    # Teste ADF
    print("Teste ADF (Augmented Dickey-Fuller):")
    try:
        adf_result = adfuller(retornos, autolag='AIC')
        adf_estatistica = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_valores_criticos = adf_result[4]

        print(f"✓ Estatística: {adf_estatistica:.6f}")
        print(f"✓ P-valor: {adf_pvalue:.6f}")
        print(f"✓ Valores críticos: {adf_valores_criticos}")

        if adf_pvalue <= 0.05:
            print("   → Rejeita H0: Série é ESTACIONÁRIA (p <= 0.05)")
        else:
            print("   → Não rejeita H0: Série é NÃO ESTACIONÁRIA (p > 0.05)")
    except Exception as e:
        print(f"❌ Erro no teste ADF: {str(e)}")
        adf_estatistica = adf_pvalue = adf_valores_criticos = None

    # Teste KPSS
    print("\nTeste KPSS (Kwiatkowski-Phillips-Schmidt-Shin):")
    try:
        kpss_result = kpss(retornos, regression='c', nlags='auto')
        kpss_estatistica = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        kpss_valores_criticos = kpss_result[3]

        print(f"✓ Estatística: {kpss_estatistica:.6f}")
        print(f"✓ P-valor: {kpss_pvalue:.6f}")
        print(f"✓ Valores críticos: {kpss_valores_criticos}")

        if kpss_pvalue <= 0.05:
            print("   → Rejeita H0: Série é NÃO ESTACIONÁRIA (p <= 0.05)")
        else:
            print("   → Não rejeita H0: Série é ESTACIONÁRIA (p > 0.05)")
    except Exception as e:
        print(f"❌ Erro no teste KPSS: {str(e)}")
        kpss_estatistica = kpss_pvalue = kpss_valores_criticos = None

    return {
        'adf_estatistica': adf_estatistica,
        'adf_pvalue': adf_pvalue,
        'kpss_estatistica': kpss_estatistica,
        'kpss_pvalue': kpss_pvalue
    }

# =============================================================================
# BLOCO 12: DOCUMENTAÇÃO DOS RESULTADOS (EAP 1.4.3)
# =============================================================================

def documentar_resultados_estacionariedade(resultado_precos, resultado_retornos):
    """
    EAP 1.4.3 - Documentar os resultados dos testes de estacionariedade
    """
    print("\n" + "=" * 60)
    print("EAP 1.4.3 - DOCUMENTAÇÃO DOS RESULTADOS")
    print("=" * 60)

    print("""
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
    """)

    # Resultados específicos
    if resultado_precos and resultado_precos['adf_pvalue'] is not None:
        print(f"RESULTADOS ESPECÍFICOS - PREÇOS:")
        print(f"✓ ADF p-valor: {resultado_precos['adf_pvalue']:.6f}")
        if resultado_precos['kpss_pvalue'] is not None:
            print(f"✓ KPSS p-valor: {resultado_precos['kpss_pvalue']:.6f}")

    if resultado_retornos and resultado_retornos['adf_pvalue'] is not None:
        print(f"\nRESULTADOS ESPECÍFICOS - RETORNOS:")
        print(f"✓ ADF p-valor: {resultado_retornos['adf_pvalue']:.6f}")
        if resultado_retornos['kpss_pvalue'] is not None:
            print(f"✓ KPSS p-valor: {resultado_retornos['kpss_pvalue']:.6f}")

# =============================================================================
# BLOCO 13: EXECUÇÃO COMPLETA DA FASE 1
# =============================================================================

def executar_fase_1_completa(caminho_arquivo, usar_dados_exemplo=False):
    """
    Executa toda a Fase 1 do EAP de forma sequencial
    """
    print("=" * 80)
    print("EXECUÇÃO COMPLETA DA FASE 1 - EAP 1.0")
    print("PROJETO: PREVISÃO DE TENDÊNCIA DO IBOVESPA")
    print("=" * 80)

    try:
        # Se usar dados de exemplo, cria primeiro
        if usar_dados_exemplo:
            print("Criando dados de exemplo...")
            criar_dados_exemplo_ibovespa()
            caminho_arquivo = 'dados_ibovespa_exemplo.csv'

        # Executa todos os blocos sequencialmente
        dados = carregar_e_validar_dados(caminho_arquivo)
        dados_limpos = limpar_e_tratar_dados(dados)
        stats = gerar_estatisticas_descritivas(dados_limpos)
        plotar_preco_e_volume(dados_limpos)
        dados_com_retornos, retornos = calcular_retornos_logaritmicos(dados_limpos)
        analisar_distribuicao_retornos(retornos)
        analisar_volatility_clustering(retornos)
        resultado_precos = testar_estacionariedade_precos(dados_com_retornos)
        resultado_retornos = testar_estacionariedade_retornos(dados_com_retornos)
        documentar_resultados_estacionariedade(resultado_precos, resultado_retornos)

        print("\n" + "=" * 80)
        print("✅ FASE 1 CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print("✓ Todos os itens do EAP 1.0 foram executados")
        print("✓ Dados validados e limpos")
        print("✓ Análise exploratória completa")
        print("✓ Testes de estacionariedade realizados")
        print("✓ Gráficos salvos")

        return dados_com_retornos, {
            'stats': stats,
            'resultado_precos': resultado_precos,
            'resultado_retornos': resultado_retornos
        }

    except Exception as e:
        print(f"\n❌ ERRO NA EXECUÇÃO: {str(e)}")
        raise

# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("FASE 1 - ARQUIVO ÚNICO PARA NOTEBOOK")
    print("=" * 50)
    print("Para usar este arquivo:")
    print("1. Execute cada bloco individualmente no notebook")
    print("2. Ou execute a função completa:")
    print("   dados, resultados = executar_fase_1_completa('seu_arquivo.csv')")
    print("3. Para dados de exemplo:")
    print("   dados, resultados = executar_fase_1_completa('', usar_dados_exemplo=True)")

    # Execução automática com dados de exemplo
    dados, resultados = executar_fase_1_completa('', usar_dados_exemplo=True)
