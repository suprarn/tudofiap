#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 4: Modelagem e Validação (EAP 4.0)
=======================================

Este módulo implementa a Fase 4 do projeto de previsão de tendência do IBOVESPA,
focando no treinamento dos modelos, na sua avaliação rigorosa e na validação 
da robustez dos resultados.

Autor: Projeto Tech Challenge 2
Data: 2025-01-24
Referência: EAP.md e Steering.md
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelagemValidacao:
    """
    Classe responsável pela modelagem e validação conforme especificado no EAP 4.0.
    """
    
    def __init__(self, dados_preparados=None):
        """
        Inicializa a classe com os dados preparados da Fase 3.
        
        Parâmetros:
        -----------
        dados_preparados : dict
            Dicionário com dados preparados da Fase 3
        """
        self.dados_preparados = dados_preparados
        self.modelo_baseline = None
        self.modelo_xgboost = None
        self.resultados_baseline = {}
        self.resultados_xgboost = {}
        self.resultados_walk_forward = {}
        
    def treinar_modelo_baseline(self):
        """
        4.1.1 - Instanciar e treinar um modelo de Regressão Logística com os dados de treino.
        4.1.2 - Realizar previsões no conjunto de teste.
        
        Retorna:
        --------
        dict
            Resultados do modelo baseline
        """
        if self.dados_preparados is None:
            raise ValueError("Dados preparados devem estar disponíveis")
        
        X_train = self.dados_preparados['X_train_scaled']
        y_train = self.dados_preparados['y_train']
        X_test = self.dados_preparados['X_test_scaled']
        y_test = self.dados_preparados['y_test']
        
        print("=== TREINAMENTO DO MODELO BASELINE (REGRESSÃO LOGÍSTICA) ===")
        
        # 4.1.1 - Instancia e treina Regressão Logística
        self.modelo_baseline = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'  # Adequado para datasets pequenos/médios
        )
        
        self.modelo_baseline.fit(X_train, y_train)
        
        # 4.1.2 - Realiza previsões
        y_pred_baseline = self.modelo_baseline.predict(X_test)
        y_pred_proba_baseline = self.modelo_baseline.predict_proba(X_test)[:, 1]
        
        # Armazena resultados
        self.resultados_baseline = {
            'modelo': self.modelo_baseline,
            'y_pred': y_pred_baseline,
            'y_pred_proba': y_pred_proba_baseline,
            'y_true': y_test
        }
        
        print(f"✓ Modelo de Regressão Logística treinado")
        print(f"✓ Previsões realizadas no conjunto de teste")
        print(f"✓ Amostras de treino: {len(X_train)}")
        print(f"✓ Amostras de teste: {len(X_test)}")
        
        return self.resultados_baseline
    
    def treinar_modelo_xgboost(self):
        """
        4.2.1 - Instanciar um XGBClassifier.
        4.2.2 - Configurar o hiperparâmetro scale_pos_weight para lidar com 
                o desbalanceamento de classe, se houver.
        4.2.3 - Treinar o modelo com os dados de treino.
        4.2.4 - Realizar previsões no conjunto de teste.
        
        Retorna:
        --------
        dict
            Resultados do modelo XGBoost
        """
        if self.dados_preparados is None:
            raise ValueError("Dados preparados devem estar disponíveis")
        
        X_train = self.dados_preparados['X_train_scaled']
        y_train = self.dados_preparados['y_train']
        X_test = self.dados_preparados['X_test_scaled']
        y_test = self.dados_preparados['y_test']
        
        print("=== TREINAMENTO DO MODELO PRINCIPAL (XGBOOST) ===")
        
        # Calcula scale_pos_weight para balanceamento
        contagem_classes = y_train.value_counts()
        scale_pos_weight = contagem_classes[0] / contagem_classes[1] if 1 in contagem_classes else 1
        
        print(f"✓ Scale pos weight calculado: {scale_pos_weight:.4f}")
        
        # 4.2.1 e 4.2.2 - Instancia XGBClassifier com configurações
        self.modelo_xgboost = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        # 4.2.3 - Treina o modelo
        self.modelo_xgboost.fit(X_train, y_train)
        
        # 4.2.4 - Realiza previsões
        y_pred_xgb = self.modelo_xgboost.predict(X_test)
        y_pred_proba_xgb = self.modelo_xgboost.predict_proba(X_test)[:, 1]
        
        # Armazena resultados
        self.resultados_xgboost = {
            'modelo': self.modelo_xgboost,
            'y_pred': y_pred_xgb,
            'y_pred_proba': y_pred_proba_xgb,
            'y_true': y_test,
            'feature_importance': self.modelo_xgboost.feature_importances_
        }
        
        print(f"✓ Modelo XGBoost treinado")
        print(f"✓ Previsões realizadas no conjunto de teste")
        print(f"✓ Hiperparâmetros configurados para robustez")
        
        return self.resultados_xgboost
    
    def avaliar_metricas_desempenho(self, resultados, nome_modelo):
        """
        4.3.1 - Para ambos os modelos, calcular e analisar:
        * Matriz de Confusão
        * Precisão (Precision)
        * Revocação (Recall)
        * F1-Score
        
        Parâmetros:
        -----------
        resultados : dict
            Resultados do modelo (baseline ou xgboost)
        nome_modelo : str
            Nome do modelo para exibição
        
        Retorna:
        --------
        dict
            Métricas de desempenho
        """
        y_true = resultados['y_true']
        y_pred = resultados['y_pred']
        
        print(f"\n=== AVALIAÇÃO DE MÉTRICAS - {nome_modelo.upper()} ===")
        
        # Matriz de Confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Métricas
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        accuracy = (y_pred == y_true).mean()
        
        # Relatório detalhado
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metricas = {
            'matriz_confusao': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'classification_report': report
        }
        
        # Exibe resultados
        print(f"✓ Matriz de Confusão:")
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}],")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        print(f"\n✓ Métricas de Desempenho:")
        print(f"   - Acurácia: {accuracy:.4f}")
        print(f"   - Precisão: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")
        
        # Interpretação das métricas
        print(f"\n✓ Interpretação:")
        print(f"   - Precisão: {precision:.2%} das previsões de 'alta' estão corretas")
        print(f"   - Recall: {recall:.2%} dos dias de 'alta' foram identificados")
        print(f"   - F1-Score: {f1:.4f} (média harmônica de precisão e recall)")
        
        return metricas
    
    def plotar_matriz_confusao(self, resultados, nome_modelo, salvar_grafico=False):
        """
        Plota a matriz de confusão de forma visual.
        
        Parâmetros:
        -----------
        resultados : dict
            Resultados do modelo
        nome_modelo : str
            Nome do modelo
        salvar_grafico : bool
            Se True, salva o gráfico
        """
        y_true = resultados['y_true']
        y_pred = resultados['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Baixa (0)', 'Alta (1)'],
                   yticklabels=['Baixa (0)', 'Alta (1)'])
        plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=14, fontweight='bold')
        plt.xlabel('Predição', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        
        if salvar_grafico:
            plt.savefig(f'Tech Challenge 2/final/matriz_confusao_{nome_modelo.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Matriz de confusão salva para {nome_modelo}")
        
        plt.show()
    
    def validacao_walk_forward(self, n_splits=3):
        """
        4.4.1 - Implementar a estrutura de validação walk-forward simplificada 
                com 3 dobras sobre os dados de teste.
        4.4.2 - Para cada dobra, treinar o modelo XGBoost com todos os dados 
                anteriores e testar no período seguinte.
        4.4.3 - Coletar as métricas de desempenho para cada dobra.
        4.4.4 - Calcular a média e o desvio padrão das métricas obtidas.
        
        Parâmetros:
        -----------
        n_splits : int
            Número de dobras para validação walk-forward
        
        Retorna:
        --------
        dict
            Resultados da validação walk-forward
        """
        if self.dados_preparados is None:
            raise ValueError("Dados preparados devem estar disponíveis")
        
        print(f"=== VALIDAÇÃO WALK-FORWARD ({n_splits} DOBRAS) ===")
        
        # Dados completos (treino + teste)
        X_completo = pd.concat([
            self.dados_preparados['X_train_scaled'], 
            self.dados_preparados['X_test_scaled']
        ])
        y_completo = pd.concat([
            self.dados_preparados['y_train'], 
            self.dados_preparados['y_test']
        ])
        
        # Configuração do TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metricas_dobras = []
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X_completo)):
            print(f"\n--- Dobra {i+1}/{n_splits} ---")
            
            # Dados da dobra
            X_train_fold = X_completo.iloc[train_idx]
            y_train_fold = y_completo.iloc[train_idx]
            X_test_fold = X_completo.iloc[test_idx]
            y_test_fold = y_completo.iloc[test_idx]
            
            # Calcula scale_pos_weight para a dobra
            contagem_classes = y_train_fold.value_counts()
            scale_pos_weight = contagem_classes[0] / contagem_classes[1] if 1 in contagem_classes else 1
            
            # Treina modelo XGBoost para a dobra
            modelo_fold = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            
            modelo_fold.fit(X_train_fold, y_train_fold)
            
            # Previsões
            y_pred_fold = modelo_fold.predict(X_test_fold)
            
            # Métricas da dobra
            precision_fold = precision_score(y_test_fold, y_pred_fold, average='binary')
            recall_fold = recall_score(y_test_fold, y_pred_fold, average='binary')
            f1_fold = f1_score(y_test_fold, y_pred_fold, average='binary')
            accuracy_fold = (y_pred_fold == y_test_fold).mean()
            
            metricas_fold = {
                'dobra': i+1,
                'precision': precision_fold,
                'recall': recall_fold,
                'f1_score': f1_fold,
                'accuracy': accuracy_fold,
                'periodo_treino': f"{X_train_fold.index.min()} até {X_train_fold.index.max()}",
                'periodo_teste': f"{X_test_fold.index.min()} até {X_test_fold.index.max()}",
                'amostras_treino': len(X_train_fold),
                'amostras_teste': len(X_test_fold)
            }
            
            metricas_dobras.append(metricas_fold)
            
            print(f"   Período treino: {metricas_fold['periodo_treino']}")
            print(f"   Período teste: {metricas_fold['periodo_teste']}")
            print(f"   F1-Score: {f1_fold:.4f}")
            print(f"   Precisão: {precision_fold:.4f}")
            print(f"   Recall: {recall_fold:.4f}")
        
        # Calcula estatísticas agregadas
        df_metricas = pd.DataFrame(metricas_dobras)
        
        estatisticas_agregadas = {
            'precision_media': df_metricas['precision'].mean(),
            'precision_std': df_metricas['precision'].std(),
            'recall_media': df_metricas['recall'].mean(),
            'recall_std': df_metricas['recall'].std(),
            'f1_score_media': df_metricas['f1_score'].mean(),
            'f1_score_std': df_metricas['f1_score'].std(),
            'accuracy_media': df_metricas['accuracy'].mean(),
            'accuracy_std': df_metricas['accuracy'].std()
        }
        
        self.resultados_walk_forward = {
            'metricas_por_dobra': metricas_dobras,
            'estatisticas_agregadas': estatisticas_agregadas,
            'dataframe_metricas': df_metricas
        }
        
        print(f"\n=== RESULTADOS AGREGADOS DA VALIDAÇÃO WALK-FORWARD ===")
        print(f"✓ F1-Score: {estatisticas_agregadas['f1_score_media']:.4f} ± {estatisticas_agregadas['f1_score_std']:.4f}")
        print(f"✓ Precisão: {estatisticas_agregadas['precision_media']:.4f} ± {estatisticas_agregadas['precision_std']:.4f}")
        print(f"✓ Recall: {estatisticas_agregadas['recall_media']:.4f} ± {estatisticas_agregadas['recall_std']:.4f}")
        print(f"✓ Acurácia: {estatisticas_agregadas['accuracy_media']:.4f} ± {estatisticas_agregadas['accuracy_std']:.4f}")
        
        return self.resultados_walk_forward
