#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Principal - Execução da Fase 1 do Projeto IBOVESPA
=========================================================

Este script demonstra a execução completa da Fase 1 conforme especificado no EAP.
Implementa todos os itens de 1.1 a 1.4 com dados de exemplo.

Autor: Projeto Tech Challenge 2
Data: 2025-01-24
Referência: EAP.md e Steering.md
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Adiciona o diretório atual ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fase_1_integridade_dados_eda import IntegridadeDadosEDA

def criar_dados_exemplo_ibovespa(n_anos=15, salvar_arquivo=True):
    """
    Cria um dataset de exemplo do IBOVESPA para demonstração.
    
    Parâmetros:
    -----------
    n_anos : int
        Número de anos de dados para gerar
    salvar_arquivo : bool
        Se True, salva o arquivo CSV
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame com dados simulados do IBOVESPA
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
    
    # Simula volume
    volume_base = 2000000000  # 2 bilhões (volume típico)
    volume = np.random.lognormal(np.log(volume_base), 0.3, n_dias)
    
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
    
    if salvar_arquivo:
        caminho_arquivo = 'dados_ibovespa_exemplo.csv'
        dados_ibovespa.to_csv(caminho_arquivo, index=False)
        print(f"✓ Arquivo salvo: {caminho_arquivo}")
    
    return dados_ibovespa

def executar_demonstracao_fase1():
    """
    Executa uma demonstração completa da Fase 1 do EAP.
    """
    print("=" * 80)
    print("DEMONSTRAÇÃO COMPLETA DA FASE 1 - EAP 1.0")
    print("PROJETO: PREVISÃO DE TENDÊNCIA DO IBOVESPA")
    print("=" * 80)
    
    try:
        # 1. Cria dados de exemplo
        print("\n" + "=" * 60)
        print("PREPARAÇÃO: CRIANDO DADOS DE EXEMPLO")
        print("=" * 60)
        
        dados_exemplo = criar_dados_exemplo_ibovespa(n_anos=15, salvar_arquivo=True)
        caminho_dados = 'dados_ibovespa_exemplo.csv'
        
        # 2. Executa Fase 1 completa
        print("\n" + "=" * 60)
        print("EXECUTANDO FASE 1 COMPLETA")
        print("=" * 60)
        
        # Instancia a classe
        eda = IntegridadeDadosEDA()
        
        # Executa todas as etapas da Fase 1
        resultados_fase1 = eda.executar_fase_1_completa(
            caminho_dados=caminho_dados,
            salvar_graficos=True
        )
        
        # 3. Resumo dos resultados
        print("\n" + "=" * 60)
        print("RESUMO DOS RESULTADOS DA FASE 1")
        print("=" * 60)
        
        if resultados_fase1['status'] == 'SUCESSO':
            print("✅ FASE 1 EXECUTADA COM SUCESSO!")
            print("\n📊 Principais Achados:")
            
            # Estatísticas dos dados
            if 'verificacao_integridade' in resultados_fase1:
                integridade = resultados_fase1['verificacao_integridade']
                print(f"   • Duplicatas encontradas: {integridade.get('duplicatas_indice', 0)}")
                print(f"   • Inconsistências de preço: {integridade.get('total_inconsistencias', 0)}")
            
            # Lacunas temporais
            if 'lacunas_temporais' in resultados_fase1:
                lacunas = resultados_fase1['lacunas_temporais']
                print(f"   • Dias úteis ausentes (feriados): {lacunas.get('dias_uteis_ausentes', 0)}")
                print(f"   • Forward fill aplicado: {lacunas.get('total_datas_ausentes', 0)} datas")
            
            # Testes de estacionariedade
            if 'teste_estacionariedade_precos' in resultados_fase1:
                teste_precos = resultados_fase1['teste_estacionariedade_precos']
                if teste_precos and 'adf_pvalue' in teste_precos:
                    print(f"   • Preços são estacionários: {'Não' if teste_precos['adf_pvalue'] > 0.05 else 'Sim'}")
            
            if 'teste_estacionariedade_retornos' in resultados_fase1:
                teste_retornos = resultados_fase1['teste_estacionariedade_retornos']
                if teste_retornos and 'adf_pvalue' in teste_retornos:
                    print(f"   • Retornos são estacionários: {'Sim' if teste_retornos['adf_pvalue'] <= 0.05 else 'Não'}")
            
            print("\n📁 Arquivos Gerados:")
            print("   • dados_ibovespa_exemplo.csv - Dataset de exemplo")
            print("   • grafico_preco_volume.png - Gráficos de preço e volume")
            print("   • histograma_retornos.png - Análise da distribuição dos retornos")
            print("   • serie_retornos_volatilidade.png - Análise de volatility clustering")
            
            print("\n🎯 Próximos Passos:")
            print("   1. Executar Fase 2: Definição do Alvo e Engenharia de Atributos")
            print("   2. Executar Fase 3: Preparação da Base para Modelagem")
            print("   3. Executar Fase 4: Modelagem e Validação")
            print("   4. Executar Fase 5: Análise, Conclusão e Relatório")
            
        else:
            print("❌ ERRO NA EXECUÇÃO DA FASE 1")
            if 'erro' in resultados_fase1:
                print(f"Erro: {resultados_fase1['erro']}")
        
        return resultados_fase1
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NA DEMONSTRAÇÃO: {str(e)}")
        print("Verifique se todas as dependências estão instaladas:")
        print("pip install pandas numpy matplotlib seaborn statsmodels scipy")
        raise

def executar_etapas_individuais():
    """
    Demonstra como executar etapas individuais da Fase 1.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO: EXECUÇÃO DE ETAPAS INDIVIDUAIS")
    print("=" * 60)
    
    # Cria dados de exemplo se não existirem
    caminho_dados = 'dados_ibovespa_exemplo.csv'
    if not os.path.exists(caminho_dados):
        criar_dados_exemplo_ibovespa(salvar_arquivo=True)
    
    # Instancia a classe
    eda = IntegridadeDadosEDA()
    
    print("\n1. Carregando dados...")
    eda.carregar_dados(caminho_dados)
    
    print("\n2. Convertendo data para índice...")
    eda.converter_data_indice()
    
    print("\n3. Verificando integridade...")
    integridade = eda.verificar_integridade()
    
    print("\n4. Identificando lacunas temporais...")
    lacunas = eda.identificar_lacunas_temporais()
    
    print("\n5. Aplicando forward fill...")
    eda.aplicar_forward_fill()
    
    print("\n6. Documentando decisão sobre outliers...")
    eda.documentar_decisao_outliers()
    
    print("\n7. Gerando estatísticas descritivas...")
    stats = eda.gerar_estatisticas_descritivas()
    
    print("\n8. Calculando retornos logarítmicos...")
    retornos = eda.calcular_retornos_logaritmicos()
    
    print("\n9. Testando estacionariedade dos preços...")
    teste_precos = eda.teste_estacionariedade_precos()
    
    print("\n10. Testando estacionariedade dos retornos...")
    teste_retornos = eda.teste_estacionariedade_retornos()
    
    print("\n11. Documentando resultados...")
    doc = eda.documentar_resultados_estacionariedade()
    
    print("\n✅ Todas as etapas individuais executadas com sucesso!")

def main():
    """
    Função principal do script.
    """
    print("SCRIPT PRINCIPAL - EXECUÇÃO DA FASE 1")
    print("Escolha uma opção:")
    print("1. Execução completa da Fase 1 (recomendado)")
    print("2. Execução de etapas individuais")
    print("3. Apenas criar dados de exemplo")
    
    try:
        opcao = input("\nDigite sua opção (1, 2 ou 3): ").strip()
        
        if opcao == "1":
            resultados = executar_demonstracao_fase1()
            return resultados
        elif opcao == "2":
            executar_etapas_individuais()
        elif opcao == "3":
            criar_dados_exemplo_ibovespa(salvar_arquivo=True)
        else:
            print("Opção inválida. Executando demonstração completa...")
            resultados = executar_demonstracao_fase1()
            return resultados
            
    except KeyboardInterrupt:
        print("\n\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"\nErro durante execução: {str(e)}")
        print("Executando demonstração completa como fallback...")
        resultados = executar_demonstracao_fase1()
        return resultados

if __name__ == "__main__":
    main()
