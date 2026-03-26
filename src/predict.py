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
# CHARGEMENT DES MODÈLES
# ============================================================
def charger_modeles():
    scaler = joblib.load('models/scaler.pkl')
    rf     = joblib.load('models/random_forest.pkl')
    kmeans = joblib.load('models/kmeans.pkl')
    lr     = joblib.load('models/linear_regression.pkl')
    pca    = joblib.load('models/pca.pkl')
    print("✅ Modèles chargés : scaler, random_forest, kmeans, linear_regression, pca")
    return scaler, rf, kmeans, lr, pca

# ============================================================
# PRÉDICTION CHURN — Classification
# ============================================================
def predire_churn(client_df, rf):
    X_train = pd.read_csv('data/train_test/X_train.csv')
    X_train = X_train.drop(columns=COLS_LEAKAGE, errors='ignore')
    client_df = client_df.drop(columns=COLS_LEAKAGE, errors='ignore')

    scaler_clf = StandardScaler()
    scaler_clf.fit(X_train)
    client_scaled = scaler_clf.transform(client_df)

    prediction  = rf.predict(client_scaled)[0]
    probabilite = rf.predict_proba(client_scaled)[0]

    return {
        'churn_predit' : int(prediction),
        'label'        : 'Churner ⚠️' if prediction == 1 else 'Fidèle ✅',
        'prob_fidele'  : round(probabilite[0] * 100, 1),
        'prob_churner' : round(probabilite[1] * 100, 1),
    }

# ============================================================
# PRÉDICTION SEGMENT — Clustering
# ============================================================
def predire_segment(client_df, scaler, kmeans):
    client_scaled = scaler.transform(client_df)
    segment = kmeans.predict(client_scaled)[0]

    labels = {
        0: 'Segment A — Clients actifs',
        1: 'Segment B — Clients réguliers',
        2: 'Segment C — Clients occasionnels',
        3: 'Segment D — Clients inactifs',
    }
    return {
        'segment_id'   : int(segment),
        'segment_label': labels.get(segment, f'Segment {segment}')
    }

# ============================================================
# PIPELINE COMPLET — Prédire sur X_test
# ============================================================
def predire_sur_test():
    print("\n" + "="*55)
    print("   PRÉDICTIONS SUR X_TEST")
    print("="*55)

    X_test = pd.read_csv('data/train_test/X_test.csv')
    y_test = pd.read_csv('data/train_test/y_test.csv').squeeze()
    X_train = pd.read_csv('data/train_test/X_train.csv')

    scaler, rf, kmeans, lr, pca = charger_modeles()

    # Supprimer leakage
    X_test_clf  = X_test.drop(columns=COLS_LEAKAGE, errors='ignore')
    X_train_clf = X_train.drop(columns=COLS_LEAKAGE, errors='ignore')

    # Refitter scaler sans les colonnes leakage
    scaler_clf = StandardScaler()
    scaler_clf.fit(X_train_clf)
    X_test_scaled = scaler_clf.transform(X_test_clf)

    predictions  = rf.predict(X_test_scaled)
    probabilites = rf.predict_proba(X_test_scaled)

    resultats = pd.DataFrame({
        'Churn_Réel'   : y_test.values,
        'Churn_Prédit' : predictions,
        'Prob_Fidèle'  : (probabilites[:, 0] * 100).round(1),
        'Prob_Churner' : (probabilites[:, 1] * 100).round(1),
    })

    print(f"\n📊 Aperçu des 10 premières prédictions :")
    print(resultats.head(10).to_string(index=False))

    correct = (resultats['Churn_Réel'] == resultats['Churn_Prédit']).sum()
    total   = len(resultats)
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
    print("   EXEMPLE — PRÉDICTION CLIENT FICTIF")
    print("="*55)

    X_train = pd.read_csv('data/train_test/X_train.csv')
    client  = pd.DataFrame([X_train.mean()], columns=X_train.columns)

    scaler, rf, kmeans, lr, pca = charger_modeles()

    churn = predire_churn(client.copy(), rf)
    print(f"\n👤 Client fictif (valeurs moyennes) :")
    print(f"   Churn prédit  : {churn['label']}")
    print(f"   Prob. Fidèle  : {churn['prob_fidele']}%")
    print(f"   Prob. Churner : {churn['prob_churner']}%")

    segment = predire_segment(client.copy(), scaler, kmeans)
    print(f"   Segment       : {segment['segment_label']}")

    
if __name__ == "__main__":
    predire_sur_test()
    exemple_client()
    print("\n🎉 predict.py terminé !")