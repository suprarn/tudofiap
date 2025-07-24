#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 3: Preparação da Base para Modelagem (EAP 3.0)
===================================================

Este módulo implementa a Fase 3 do projeto de previsão de tendência do IBOVESPA,
estruturando os dados para que possam ser consumidos pelos algoritmos de 
machine learning, respeitando a ordem temporal.

Autor: Projeto Tech Challenge 2
Data: 2025-01-24
Referência: EAP.md e Steering.md
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class PreparacaoModelagem:
    """
    Classe responsável pela preparação dos dados para modelagem
    conforme especificado no EAP 3.0.
    """
    
    def __init__(self, dados_com_features=None):
        """
        Inicializa a classe com os dados com features da Fase 2.
        
        Parâmetros:
        -----------
        dados_com_features : pd.DataFrame
            DataFrame com features da Fase 2
        """
        self.dados_com_features = dados_com_features
        self.dados_janela_deslizante = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.janela_tamanho = 5
        
    def estruturar_janela_deslizante(self, janela_tamanho=5):
        """
        3.1.1 - Definir a janela de entrada (lookback window) como n=5 dias.
        
        3.1.2 - Implementar a lógica para transformar a série temporal em um 
        dataset tabular, onde cada linha contém os atributos dos últimos 5 dias 
        e o alvo correspondente.
        
        Parâmetros:
        -----------
        janela_tamanho : int
            Tamanho da janela deslizante (padrão: 5 dias)
        
        Retorna:
        --------
        tuple
            (X, y) onde X são as features e y são os targets
        """
        if self.dados_com_features is None:
            raise ValueError("Dados com features devem estar disponíveis")
        
        self.janela_tamanho = janela_tamanho
        
        # Remove colunas que não são features (mantém apenas features numéricas)
        colunas_excluir = ['Target']
        
        # Identifica features numéricas
        features_numericas = self.dados_com_features.select_dtypes(include=[np.number]).columns
        features_numericas = [col for col in features_numericas if col not in colunas_excluir]
        
        # Remove linhas com NaN nas features
        dados_limpos = self.dados_com_features[features_numericas + ['Target']].dropna()
        
        print(f"=== ESTRUTURAÇÃO COM JANELA DESLIZANTE ===")
        print(f"✓ Tamanho da janela: {janela_tamanho} dias")
        print(f"✓ Features selecionadas: {len(features_numericas)}")
        print(f"✓ Dados limpos: {len(dados_limpos)} registros")
        
        # Cria estrutura de janela deslizante
        X_list = []
        y_list = []
        indices_list = []
        
        for i in range(janela_tamanho, len(dados_limpos)):
            # Janela de features (últimos n dias)
            janela_features = dados_limpos[features_numericas].iloc[i-janela_tamanho:i].values
            
            # Achata a janela (transforma matriz em vetor)
            janela_achatada = janela_features.flatten()
            
            # Target correspondente
            target = dados_limpos['Target'].iloc[i]
            
            X_list.append(janela_achatada)
            y_list.append(target)
            indices_list.append(dados_limpos.index[i])
        
        # Converte para arrays numpy
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Cria nomes das colunas para o DataFrame final
        nomes_colunas = []
        for lag in range(janela_tamanho, 0, -1):
            for feature in features_numericas:
                nomes_colunas.append(f"{feature}_lag_{lag}")
        
        # Cria DataFrame estruturado
        self.dados_janela_deslizante = pd.DataFrame(
            X, 
            columns=nomes_colunas,
            index=indices_list
        )
        self.dados_janela_deslizante['Target'] = y
        
        print(f"✓ Dataset estruturado criado: {X.shape[0]} amostras, {X.shape[1]} features")
        print(f"✓ Cada amostra contém {janela_tamanho} dias de {len(features_numericas)} features")
        
        return X, y
    
    def divisao_cronologica_dados(self, proporcao_treino=0.8):
        """
        3.2.1 - Definir uma data de corte para a divisão treino-teste 
        (ex: 80% para treino, 20% para teste).
        
        3.2.2 - Separar os dados em X_train, y_train, X_test, e y_test 
        sem usar amostragem aleatória.
        
        Parâmetros:
        -----------
        proporcao_treino : float
            Proporção dos dados para treinamento (padrão: 0.8)
        
        Retorna:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.dados_janela_deslizante is None:
            raise ValueError("Dados com janela deslizante devem estar disponíveis")
        
        # Calcula ponto de corte cronológico
        total_amostras = len(self.dados_janela_deslizante)
        ponto_corte = int(total_amostras * proporcao_treino)
        
        # Data de corte
        data_corte = self.dados_janela_deslizante.index[ponto_corte]
        
        # Separação cronológica
        dados_treino = self.dados_janela_deslizante.iloc[:ponto_corte]
        dados_teste = self.dados_janela_deslizante.iloc[ponto_corte:]
        
        # Separa features e target
        colunas_features = [col for col in self.dados_janela_deslizante.columns if col != 'Target']
        
        self.X_train = dados_treino[colunas_features]
        self.y_train = dados_treino['Target']
        self.X_test = dados_teste[colunas_features]
        self.y_test = dados_teste['Target']
        
        print(f"=== DIVISÃO CRONOLÓGICA DOS DADOS ===")
        print(f"✓ Proporção treino/teste: {proporcao_treino:.0%}/{1-proporcao_treino:.0%}")
        print(f"✓ Data de corte: {data_corte}")
        print(f"✓ Período de treino: {self.X_train.index.min()} até {self.X_train.index.max()}")
        print(f"✓ Período de teste: {self.X_test.index.min()} até {self.X_test.index.max()}")
        print(f"✓ Amostras de treino: {len(self.X_train)}")
        print(f"✓ Amostras de teste: {len(self.X_test)}")
        print(f"✓ Features: {len(colunas_features)}")
        
        # Verifica distribuição de classes
        dist_treino = self.y_train.value_counts(normalize=True)
        dist_teste = self.y_test.value_counts(normalize=True)
        
        print(f"\n✓ Distribuição de classes no treino:")
        for classe, prop in dist_treino.items():
            print(f"   - Classe {classe}: {prop:.2%}")
        
        print(f"✓ Distribuição de classes no teste:")
        for classe, prop in dist_teste.items():
            print(f"   - Classe {classe}: {prop:.2%}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def escalonamento_atributos(self):
        """
        3.3.1 - Instanciar um StandardScaler.
        
        3.3.2 - Ajustar (fit) o scaler exclusivamente no conjunto de treino (X_train).
        
        3.3.3 - Aplicar a transformação (transform) nos conjuntos de treino e teste.
        
        Retorna:
        --------
        tuple
            (X_train_scaled, X_test_scaled)
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Divisão treino/teste deve ser realizada primeiro")
        
        # 3.3.1 - Instancia StandardScaler
        self.scaler = StandardScaler()
        
        # 3.3.2 - Ajusta APENAS no conjunto de treino
        self.scaler.fit(self.X_train)
        
        # 3.3.3 - Aplica transformação em ambos os conjuntos
        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Converte de volta para DataFrame mantendo índices e nomes das colunas
        self.X_train_scaled = pd.DataFrame(
            X_train_scaled,
            index=self.X_train.index,
            columns=self.X_train.columns
        )
        
        self.X_test_scaled = pd.DataFrame(
            X_test_scaled,
            index=self.X_test.index,
            columns=self.X_test.columns
        )
        
        print(f"=== ESCALONAMENTO DE ATRIBUTOS ===")
        print(f"✓ StandardScaler instanciado e ajustado")
        print(f"✓ Scaler ajustado APENAS nos dados de treino")
        print(f"✓ Transformação aplicada em treino e teste")
        print(f"✓ Estatísticas do scaler (baseadas no treino):")
        print(f"   - Média das features: {self.scaler.mean_[:5]}")  # Mostra apenas 5 primeiras
        print(f"   - Desvio padrão das features: {self.scaler.scale_[:5]}")
        
        # Verifica se a transformação foi aplicada corretamente
        print(f"\n✓ Verificação da transformação:")
        print(f"   - Média do treino escalonado: {self.X_train_scaled.mean().mean():.6f} (deve ser ~0)")
        print(f"   - Desvio padrão do treino escalonado: {self.X_train_scaled.std().mean():.6f} (deve ser ~1)")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def executar_fase_3_completa(self, dados_com_features, janela_tamanho=5, proporcao_treino=0.8):
        """
        Executa toda a Fase 3 do EAP de forma sequencial.
        
        Parâmetros:
        -----------
        dados_com_features : pd.DataFrame
            DataFrame com features da Fase 2
        janela_tamanho : int
            Tamanho da janela deslizante
        proporcao_treino : float
            Proporção dos dados para treinamento
        
        Retorna:
        --------
        dict
            Dados preparados para modelagem
        """
        print("=" * 80)
        print("INICIANDO FASE 3: PREPARAÇÃO DA BASE PARA MODELAGEM")
        print("=" * 80)
        
        self.dados_com_features = dados_com_features
        
        try:
            # 3.1 - Estruturação com Janela Deslizante
            print("\n" + "=" * 50)
            print("3.1 - ESTRUTURAÇÃO COM JANELA DESLIZANTE")
            print("=" * 50)
            
            X, y = self.estruturar_janela_deslizante(janela_tamanho)
            
            # 3.2 - Divisão Cronológica dos Dados
            print("\n" + "=" * 50)
            print("3.2 - DIVISÃO CRONOLÓGICA DOS DADOS")
            print("=" * 50)
            
            X_train, X_test, y_train, y_test = self.divisao_cronologica_dados(proporcao_treino)
            
            # 3.3 - Escalonamento de Atributos
            print("\n" + "=" * 50)
            print("3.3 - ESCALONAMENTO DE ATRIBUTOS")
            print("=" * 50)
            
            X_train_scaled, X_test_scaled = self.escalonamento_atributos()
            
            # Resumo final
            print("\n" + "=" * 80)
            print("FASE 3 CONCLUÍDA COM SUCESSO!")
            print("=" * 80)
            print("✓ Dados estruturados com janela deslizante")
            print("✓ Divisão cronológica realizada")
            print("✓ Escalonamento aplicado corretamente")
            print("✓ Dados prontos para modelagem")
            
            # Retorna dados preparados
            dados_preparados = {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'X_train_scaled': self.X_train_scaled,
                'X_test_scaled': self.X_test_scaled,
                'scaler': self.scaler,
                'janela_tamanho': self.janela_tamanho,
                'dados_janela_deslizante': self.dados_janela_deslizante
            }
            
            return dados_preparados
            
        except Exception as e:
            print(f"\n❌ ERRO NA EXECUÇÃO DA FASE 3: {str(e)}")
            raise


def main():
    """
    Função principal para demonstração da Fase 3.
    """
    print("DEMONSTRAÇÃO DA FASE 3 - PREPARAÇÃO DA BASE PARA MODELAGEM")
    print("=" * 70)
    
    print("Para executar a Fase 3 completa, use:")
    print("prep = PreparacaoModelagem()")
    print("dados_preparados = prep.executar_fase_3_completa(dados_com_features)")
    print("\nOu execute cada etapa individualmente conforme documentado no EAP.")


if __name__ == "__main__":
    main()
