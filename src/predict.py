import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLS_LEAKAGE = ['ChurnRiskCategory', 'CustomerType_Perdu']

# ============================================================
# CHARGEMENT DES MODÈLES (version finale)
# ============================================================
def charger_modeles():
    scaler = joblib.load('models/scaler.pkl')
    rf = joblib.load('models/random_forest.pkl')
    xgb = joblib.load('models/xgboost.pkl')
    stacking = joblib.load('models/stacking.pkl')
    kmeans = joblib.load('models/kmeans.pkl')
    pca = joblib.load('models/pca.pkl')
    # Régression XGBoost (au lieu de linéaire)
    reg_model = joblib.load('models/regression_xgboost_optimized.pkl')
    reg_scaler = joblib.load('models/scaler_regression.pkl')
    print("✅ Modèles chargés : scaler, random_forest, xgboost, stacking, kmeans, pca, regression_xgboost")
    return scaler, rf, xgb, stacking, kmeans, pca, reg_model, reg_scaler

# ============================================================
# PRÉDICTION CHURN — Random Forest (pour compatibilité)
# ============================================================
def predire_churn_rf(client_df, rf):
    X_train = pd.read_csv('data/train_test/X_train.csv')
    X_train = X_train.drop(columns=COLS_LEAKAGE, errors='ignore')
    client_df = client_df.drop(columns=COLS_LEAKAGE, errors='ignore')
    scaler_clf = StandardScaler()
    scaler_clf.fit(X_train)
    client_scaled = scaler_clf.transform(client_df)
    prediction = rf.predict(client_scaled)[0]
    probabilite = rf.predict_proba(client_scaled)[0]
    return {
        'churn_predit': int(prediction),
        'label': 'Churner ⚠️' if prediction == 1 else 'Fidèle ✅',
        'prob_fidele': round(probabilite[0]*100, 1),
        'prob_churner': round(probabilite[1]*100, 1),
    }

# ============================================================
# PRÉDICTION SEGMENT — Clustering (avec PCA)
# ============================================================
def predire_segment(client_df, scaler, pca, kmeans):
    client_scaled = scaler.transform(client_df)
    client_pca = pca.transform(client_scaled)
    segment = kmeans.predict(client_pca)[0]
    labels = {
        0: 'Segment A — Clients Premium',
        1: 'Segment B — Clients Réguliers',
        2: 'Segment C — Clients Occasionnels',
        3: 'Segment D — Clients Inactifs',
    }
    return {
        'segment_id': int(segment),
        'segment_label': labels.get(segment, f'Segment {segment}')
    }

# ============================================================
# PIPELINE COMPLET — Prédire sur X_test (avec Random Forest)
# ============================================================
def predire_sur_test():
    print("\n" + "="*55)
    print("   PRÉDICTIONS SUR X_TEST (Random Forest)")
    print("="*55)
    X_test = pd.read_csv('data/train_test/X_test.csv')
    y_test = pd.read_csv('data/train_test/y_test.csv').squeeze()
    X_train = pd.read_csv('data/train_test/X_train.csv')
    scaler, rf, xgb, stacking, kmeans, pca, reg_model, reg_scaler = charger_modeles()
    # Supprimer leakage
    X_test_clf = X_test.drop(columns=COLS_LEAKAGE, errors='ignore')
    X_train_clf = X_train.drop(columns=COLS_LEAKAGE, errors='ignore')
    scaler_clf = StandardScaler()
    scaler_clf.fit(X_train_clf)
    X_test_scaled = scaler_clf.transform(X_test_clf)
    predictions = rf.predict(X_test_scaled)
    probabilites = rf.predict_proba(X_test_scaled)
    resultats = pd.DataFrame({
        'Churn_Réel': y_test.values,
        'Churn_Prédit': predictions,
        'Prob_Fidèle': (probabilites[:,0]*100).round(1),
        'Prob_Churner': (probabilites[:,1]*100).round(1),
    })
    print(f"\n📊 Aperçu des 10 premières prédictions :")
    print(resultats.head(10).to_string(index=False))
    correct = (resultats['Churn_Réel'] == resultats['Churn_Prédit']).sum()
    total = len(resultats)
    print(f"\n✅ Prédictions correctes : {correct}/{total} ({correct/total*100:.1f}%)")
    os.makedirs('reports', exist_ok=True)
    resultats.to_csv('reports/predictions_test.csv', index=False)
    print(f"✅ Résultats sauvegardés : reports/predictions_test.csv")
    return resultats

# ============================================================
# EXEMPLE — Prédire pour UN client fictif
# ============================================================
def exemple_client():
    print("\n" + "="*55)
    print("   EXEMPLE — PRÉDICTION CLIENT FICTIF (avec Random Forest et PCA)")
    print("="*55)
    X_train = pd.read_csv('data/train_test/X_train.csv')
    client = pd.DataFrame([X_train.mean()], columns=X_train.columns)
    scaler, rf, xgb, stacking, kmeans, pca, reg_model, reg_scaler = charger_modeles()
    churn = predire_churn_rf(client.copy(), rf)
    print(f"\n👤 Client fictif (valeurs moyennes) :")
    print(f"   Churn prédit  : {churn['label']}")
    print(f"   Prob. Fidèle  : {churn['prob_fidele']}%")
    print(f"   Prob. Churner : {churn['prob_churner']}%")
    segment = predire_segment(client.copy(), scaler, pca, kmeans)
    print(f"   Segment       : {segment['segment_label']}")

if __name__ == "__main__":
    predire_sur_test()
    exemple_client()
    print("\n🎉 predict.py terminé !")