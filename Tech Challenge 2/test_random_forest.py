#!/usr/bin/env python3
"""
Teste da Fase 4.2 - Random Forest para Previsão IBOVESPA
Seguindo o EAP - Projeto Previsão IBOVESPA Alta/Baixa
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importações para machine learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Carregar e preparar dados do IBOVESPA"""
    print("=== CARREGANDO E PREPARANDO DADOS ===")
    
    # Carregar dataset
    df = pd.read_csv('Dados Históricos - Ibovespa.csv', encoding='utf-8')
    print(f"Dataset carregado: {df.shape}")
    
    # Converter data e ordenar
    df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
    df = df.sort_values('Data').reset_index(drop=True)
    
    # Tratar valores ausentes
    df['Vol.'] = df['Vol.'].fillna(method='ffill')
    
    # Converter Volume para numérico
    def converter_volume(vol_str):
        if pd.isna(vol_str): return np.nan
        vol_str = str(vol_str).replace(',', '.')
        if 'B' in vol_str: return float(vol_str.replace('B', '')) * 1e9
        elif 'M' in vol_str: return float(vol_str.replace('M', '')) * 1e6
        elif 'K' in vol_str: return float(vol_str.replace('K', '')) * 1e3
        return float(vol_str)
    
    df['Volume'] = df['Vol.'].apply(converter_volume)
    df['Variacao'] = df['Var%'].str.replace('%', '').str.replace(',', '.').astype(float) / 100
    
    # Criar target
    df['Target'] = (df['Variacao'].shift(-1) > 0).astype(int)
    df = df[:-1].copy()  # Remove última linha
    
    print(f"Target distribuição: {df['Target'].value_counts(normalize=True).to_dict()}")
    
    return df

def create_features(df):
    """Criar features técnicas"""
    print("=== CRIANDO FEATURES TÉCNICAS ===")
    
    df_features = df[['Data', 'Último', 'Abertura', 'Máxima', 'Mínima', 'Volume', 'Variacao', 'Target']].copy()
    
    # Médias móveis
    for periodo in [5, 10, 20, 50]:
        df_features[f'MA_{periodo}'] = df_features['Último'].rolling(window=periodo).mean()
    
    # Bandas de Bollinger
    ma_20 = df_features['MA_20']
    std_20 = df_features['Último'].rolling(window=20).std()
    df_features['BB_Upper'] = ma_20 + (2 * std_20)
    df_features['BB_Lower'] = ma_20 - (2 * std_20)
    df_features['BB_Width'] = df_features['BB_Upper'] - df_features['BB_Lower']
    df_features['BB_Position'] = (df_features['Último'] - df_features['BB_Lower']) / df_features['BB_Width']
    
    # RSI
    def calcular_rsi(precos, periodo=14):
        delta = precos.diff()
        ganho = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganho / perda
        return 100 - (100 / (1 + rs))
    
    df_features['RSI'] = calcular_rsi(df_features['Último'])
    
    # Features de preço
    df_features['Price_Range'] = df_features['Máxima'] - df_features['Mínima']
    df_features['Price_Position'] = (df_features['Último'] - df_features['Mínima']) / df_features['Price_Range']
    df_features['Gap'] = df_features['Abertura'] - df_features['Último'].shift(1)
    
    # Volatilidade
    df_features['high_low'] = df_features['Máxima'] - df_features['Mínima']
    df_features['high_close_prev'] = abs(df_features['Máxima'] - df_features['Último'].shift(1))
    df_features['low_close_prev'] = abs(df_features['Mínima'] - df_features['Último'].shift(1))
    df_features['true_range'] = df_features[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    
    # ATR
    for periodo in [5, 10, 20]:
        df_features[f'atr_{periodo}'] = df_features['true_range'].rolling(periodo).mean()
    
    # Volatilidade histórica
    df_features['returns'] = df_features['Último'].pct_change()
    for periodo in [5, 10, 20]:
        df_features[f'volatility_{periodo}'] = df_features['returns'].rolling(periodo).std()
    
    df_features['hl_close_ratio'] = (df_features['Máxima'] - df_features['Mínima']) / df_features['Último']
    
    # Features temporais
    df_features['day_of_week'] = df_features['Data'].dt.dayofweek
    df_features['month'] = df_features['Data'].dt.month
    df_features['quarter'] = df_features['Data'].dt.quarter
    df_features['is_month_start'] = (df_features['Data'].dt.day <= 5).astype(int)
    df_features['is_month_end'] = (df_features['Data'].dt.day >= 25).astype(int)
    
    print(f"Features criadas: {df_features.shape[1]} colunas")
    
    return df_features

def select_features(df_features):
    """Selecionar features mais relevantes"""
    print("=== SELECIONANDO FEATURES ===")
    
    # Features selecionadas (sem data leakage)
    selected_features = [
        'quarter', 'returns', 'Volume', 'volatility_20', 'low_close_prev', 
        'atr_20', 'volatility_5', 'hl_close_ratio', 'BB_Width', 'BB_Position', 
        'atr_5', 'day_of_week', 'Price_Position', 'true_range', 'volatility_10'
    ]
    
    # Remover linhas com NaN
    df_final = df_features.dropna()
    
    print(f"Dataset final: {df_final.shape}")
    print(f"Features selecionadas: {len(selected_features)}")
    
    return df_final, selected_features

def test_random_forest(df_final, selected_features):
    """Testar Random Forest"""
    print("\n" + "="*60)
    print("=== FASE 4.2 - RANDOM FOREST ===")
    print("="*60)
    
    # Preparar dados
    X = df_final[selected_features].copy()
    y = df_final['Target'].copy()
    
    print(f"Dados: X{X.shape}, y{y.shape}")
    print(f"Target distribuição: {y.value_counts(normalize=True).to_dict()}")
    
    # Configurar validação temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 1. Random Forest Básico
    print("\n--- Random Forest Básico ---")
    rf_basic = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_basic, X, y, cv=tscv, scoring='accuracy')
    
    print(f"Acurácia por fold: {[f'{score:.1%}' for score in rf_scores]}")
    print(f"Acurácia média: {rf_scores.mean():.1%} (±{rf_scores.std():.3f})")
    
    # 2. Random Forest Otimizado
    print("\n--- Random Forest Otimizado ---")
    rf_optimized = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=20, 
        min_samples_leaf=10, max_features='sqrt', bootstrap=True,
        oob_score=True, random_state=42, n_jobs=-1
    )
    
    rf_opt_scores = cross_val_score(rf_optimized, X, y, cv=tscv, scoring='accuracy')
    
    print(f"Acurácia por fold: {[f'{score:.1%}' for score in rf_opt_scores]}")
    print(f"Acurácia média: {rf_opt_scores.mean():.1%} (±{rf_opt_scores.std():.3f})")
    
    # Treinar para OOB score
    rf_optimized.fit(X, y)
    print(f"Out-of-Bag Score: {rf_optimized.oob_score_:.1%}")
    
    # 3. Comparação com baselines
    print("\n--- Comparação com Baselines ---")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Random Forest Básico': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Random Forest Otimizado': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20, 
            min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
        )
    }
    
    results = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors']:
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        else:
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        results[name] = scores.mean()
    
    print(f"{'Modelo':<25} {'Acurácia':<10}")
    print("-" * 35)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<25} {acc:.1%}")
    
    # 4. Análise de importância
    print("\n--- Top 10 Features Mais Importantes ---")
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_optimized.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:15s}: {row['importance']:.1%}")
    
    # 5. Conclusões
    best_acc = max(results.values())
    baseline_acc = results['Logistic Regression']
    improvement = best_acc - baseline_acc
    
    print(f"\n{'='*60}")
    print("=== CONCLUSÕES DA FASE 4.2 ===")
    print(f"{'='*60}")
    print(f"🏆 Melhor modelo: {max(results.items(), key=lambda x: x[1])[0]}")
    print(f"📊 Acurácia: {best_acc:.1%}")
    print(f"📈 Melhoria sobre baseline: +{improvement:.1%}")
    
    # Status da meta de 70%
    target_acc = 0.70
    progress = best_acc / target_acc
    
    print(f"\n🎯 PROGRESSO PARA META DE 70%:")
    print(f"   Atual: {best_acc:.1%}")
    print(f"   Meta:  {target_acc:.1%}")
    print(f"   Progresso: {progress:.1%}")
    print(f"   Faltam: {target_acc - best_acc:.1%}")
    
    if best_acc >= target_acc:
        print("   🎉 META ALCANÇADA!")
    elif best_acc >= 0.60:
        print("   🔥 Muito próximo da meta!")
    elif best_acc >= 0.55:
        print("   ✅ Critério de sucesso atingido (>55%)")
    else:
        print("   ⚠️  Necessário mais otimização")
    
    print(f"\n🚀 PRÓXIMOS PASSOS:")
    if best_acc < target_acc:
        print("   1. Implementar XGBoost")
        print("   2. Criar ensemble com Voting Classifier")
        print("   3. Adicionar features técnicas avançadas")
        print("   4. Incorporar dados externos")
    
    return results, feature_importance

if __name__ == "__main__":
    try:
        # Executar pipeline completo
        df = load_and_prepare_data()
        df_features = create_features(df)
        df_final, selected_features = select_features(df_features)
        results, feature_importance = test_random_forest(df_final, selected_features)
        
        print(f"\n✅ Fase 4.2 concluída com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()
