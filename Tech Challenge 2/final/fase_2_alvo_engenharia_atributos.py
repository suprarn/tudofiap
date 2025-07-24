#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 2: Definição do Alvo e Engenharia de Atributos (EAP 2.0)
=============================================================

Este módulo implementa a Fase 2 do projeto de previsão de tendência do IBOVESPA,
focando na criação da variável a ser prevista e no enriquecimento dos dados 
com atributos preditivos, com atenção rigorosa para evitar viés de lookahead.

Autor: Projeto Tech Challenge 2
Data: 2025-01-24
Referência: EAP.md e Steering.md
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

class AlvoEngenhariaAtributos:
    """
    Classe responsável pela criação da variável alvo e engenharia de atributos
    conforme especificado no EAP 2.0.
    """
    
    def __init__(self, dados_limpos=None):
        """
        Inicializa a classe com os dados limpos da Fase 1.
        
        Parâmetros:
        -----------
        dados_limpos : pd.DataFrame
            DataFrame com dados limpos da Fase 1
        """
        self.dados_limpos = dados_limpos
        self.dados_com_target = None
        self.dados_com_features = None
        self.dicionario_atributos = {}
        
    def criar_variavel_alvo(self):
        """
        2.1.1 - Implementar a lógica para criar a coluna Target: 
        1 se Close(t+1) > Close(t), e 0 caso contrário.
        
        2.1.2 - Aplicar o deslocamento (.shift(-1)) na coluna Target para 
        alinhar corretamente os atributos do dia t com o resultado do dia t+1.
        
        2.1.3 - Remover a última linha do DataFrame, que conterá um valor 
        nulo (NaN) no Target após o deslocamento.
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com a variável alvo criada
        """
        if self.dados_limpos is None:
            raise ValueError("Dados limpos devem ser fornecidos")
        
        # Copia os dados para não modificar o original
        self.dados_com_target = self.dados_limpos.copy()
        
        # Identifica coluna de fechamento
        col_close = None
        for col in self.dados_com_target.columns:
            if 'close' in col.lower():
                col_close = col
                break
        
        if col_close is None:
            raise ValueError("Coluna de preço de fechamento não encontrada")
        
        # 2.1.1 - Cria a variável alvo
        # Target = 1 se Close(t+1) > Close(t), 0 caso contrário
        close_amanha = self.dados_com_target[col_close].shift(-1)
        close_hoje = self.dados_com_target[col_close]
        
        self.dados_com_target['Target'] = (close_amanha > close_hoje).astype(int)
        
        # 2.1.2 - O shift(-1) já foi aplicado na criação do target
        # 2.1.3 - Remove a última linha que contém NaN no Target
        self.dados_com_target = self.dados_com_target.dropna(subset=['Target'])
        
        print("=== CRIAÇÃO DA VARIÁVEL ALVO ===")
        print(f"✓ Variável Target criada com sucesso")
        print(f"✓ Lógica: Target = 1 se Close(t+1) > Close(t), 0 caso contrário")
        print(f"✓ Registros após remoção de NaN: {len(self.dados_com_target)}")
        
        return self.dados_com_target
    
    def analisar_distribuicao_classes(self):
        """
        2.2.1 - Calcular e analisar a frequência das classes 0 e 1 na coluna Target.
        
        2.2.2 - Documentar o nível de desbalanceamento e suas implicações para 
        a seleção de métricas e treinamento do modelo.
        
        Retorna:
        --------
        dict
            Análise da distribuição de classes
        """
        if self.dados_com_target is None or 'Target' not in self.dados_com_target.columns:
            raise ValueError("Variável alvo deve ser criada primeiro")
        
        # Calcula distribuição
        distribuicao = self.dados_com_target['Target'].value_counts()
        proporcoes = self.dados_com_target['Target'].value_counts(normalize=True)
        
        # Análise de desbalanceamento
        classe_majoritaria = proporcoes.idxmax()
        prop_majoritaria = proporcoes.max()
        prop_minoritaria = proporcoes.min()
        
        razao_desbalanceamento = prop_majoritaria / prop_minoritaria
        
        resultados = {
            'distribuicao_absoluta': distribuicao.to_dict(),
            'distribuicao_percentual': proporcoes.to_dict(),
            'classe_majoritaria': classe_majoritaria,
            'proporcao_majoritaria': prop_majoritaria,
            'proporcao_minoritaria': prop_minoritaria,
            'razao_desbalanceamento': razao_desbalanceamento
        }
        
        print("=== ANÁLISE DA DISTRIBUIÇÃO DE CLASSES ===")
        print(f"✓ Distribuição absoluta:")
        for classe, count in distribuicao.items():
            print(f"   - Classe {classe}: {count} ({proporcoes[classe]:.2%})")
        
        print(f"\n✓ Análise de desbalanceamento:")
        print(f"   - Classe majoritária: {classe_majoritaria} ({prop_majoritaria:.2%})")
        print(f"   - Razão de desbalanceamento: {razao_desbalanceamento:.2f}:1")
        
        if razao_desbalanceamento > 1.5:
            print(f"   - Status: DESBALANCEADO (razão > 1.5)")
            print(f"   - Implicações:")
            print(f"     • Acurácia não é uma métrica confiável")
            print(f"     • Usar Precisão, Recall e F1-Score")
            print(f"     • Considerar scale_pos_weight no XGBoost")
        else:
            print(f"   - Status: BALANCEADO (razão <= 1.5)")
        
        return resultados
    
    def criar_atributos_momento(self):
        """
        2.3.1 - Atributos de Momento: Criar colunas de retornos defasados (lags) 
        para os últimos 5 dias (Return_Lag_1 a Return_Lag_5).
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com atributos de momento adicionados
        """
        if self.dados_com_target is None:
            raise ValueError("Dados com target devem estar disponíveis")
        
        if self.dados_com_features is None:
            self.dados_com_features = self.dados_com_target.copy()
        
        # Verifica se retornos já existem
        if 'Retornos_Log' not in self.dados_com_features.columns:
            # Calcula retornos se não existirem
            col_close = None
            for col in self.dados_com_features.columns:
                if 'close' in col.lower():
                    col_close = col
                    break
            
            if col_close is None:
                raise ValueError("Coluna de fechamento não encontrada")
            
            self.dados_com_features['Retornos_Log'] = np.log(
                self.dados_com_features[col_close] / self.dados_com_features[col_close].shift(1)
            )
        
        # Cria lags dos retornos
        for lag in range(1, 6):  # Lags de 1 a 5 dias
            col_name = f'Return_Lag_{lag}'
            self.dados_com_features[col_name] = self.dados_com_features['Retornos_Log'].shift(lag)
            
            # Adiciona ao dicionário de atributos
            self.dicionario_atributos[col_name] = {
                'calculo': f'Retorno logarítmico defasado em {lag} dia(s)',
                'dados_entrada': 'Close',
                'intuicao_financeira': f'Inércia/reversão de {"curto" if lag <= 2 else "médio"} prazo',
                'categoria': 'Momento'
            }
        
        print("=== ATRIBUTOS DE MOMENTO ===")
        print("✓ Criados retornos defasados (Return_Lag_1 a Return_Lag_5)")
        print("✓ Capturam informações sobre momento de curto e médio prazo")
        
        return self.dados_com_features
    
    def criar_atributos_tendencia(self):
        """
        2.3.2 - Atributos de Tendência:
        * Calcular as Médias Móveis Simples (SMA) de 10, 20 e 50 dias.
        * Criar atributos normalizados: Ratio_Close_SMA20 e Ratio_SMA10_SMA50.
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com atributos de tendência adicionados
        """
        if self.dados_com_features is None:
            raise ValueError("Dados com features devem estar disponíveis")
        
        # Identifica coluna de fechamento
        col_close = None
        for col in self.dados_com_features.columns:
            if 'close' in col.lower():
                col_close = col
                break
        
        if col_close is None:
            raise ValueError("Coluna de fechamento não encontrada")
        
        # Calcula SMAs
        self.dados_com_features['SMA_10'] = self.dados_com_features[col_close].rolling(window=10).mean()
        self.dados_com_features['SMA_20'] = self.dados_com_features[col_close].rolling(window=20).mean()
        self.dados_com_features['SMA_50'] = self.dados_com_features[col_close].rolling(window=50).mean()
        
        # Cria atributos normalizados
        self.dados_com_features['Ratio_Close_SMA20'] = (
            self.dados_com_features[col_close] / self.dados_com_features['SMA_20']
        )
        self.dados_com_features['Ratio_SMA10_SMA50'] = (
            self.dados_com_features['SMA_10'] / self.dados_com_features['SMA_50']
        )
        
        # Adiciona ao dicionário de atributos
        self.dicionario_atributos['Ratio_Close_SMA20'] = {
            'calculo': 'Close / SMA_20',
            'dados_entrada': 'Close',
            'intuicao_financeira': 'Mede o quão "esticado" o preço está em relação à sua média de curto prazo',
            'categoria': 'Tendência'
        }
        
        self.dicionario_atributos['Ratio_SMA10_SMA50'] = {
            'calculo': 'SMA_10 / SMA_50',
            'dados_entrada': 'Close',
            'intuicao_financeira': 'Sinaliza cruzamentos de médias, indicando mudanças de tendência',
            'categoria': 'Tendência'
        }
        
        print("=== ATRIBUTOS DE TENDÊNCIA ===")
        print("✓ Criadas SMAs de 10, 20 e 50 dias")
        print("✓ Criados ratios normalizados (Ratio_Close_SMA20, Ratio_SMA10_SMA50)")
        print("✓ Capturam informações sobre tendências de curto, médio e longo prazo")
        
        return self.dados_com_features
