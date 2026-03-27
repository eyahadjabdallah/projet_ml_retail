from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

COLS_LEAKAGE = ['ChurnRiskCategory', 'CustomerType_Perdu']

# ============================================================
# Chargement des modèles au démarrage
# ============================================================
rf     = joblib.load('models/random_forest.pkl')
kmeans = joblib.load('models/kmeans.pkl')
scaler = joblib.load('models/scaler.pkl')

X_train = pd.read_csv('data/train_test/X_train.csv')
X_train_clf = X_train.drop(columns=COLS_LEAKAGE, errors='ignore')
scaler_clf = StandardScaler()
scaler_clf.fit(X_train_clf)

# ============================================================
# PAGE PRINCIPALE
# ============================================================
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# ============================================================
# PRÉDICTION — reçoit le formulaire et prédit
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les valeurs du formulaire
        recency   = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary  = float(request.form['monetary'])
        age       = float(request.form['age'])
        tenure    = float(request.form['tenure'])

        # Créer un client avec les valeurs moyennes pour toutes les colonnes
        client = pd.DataFrame([X_train_clf.mean()], columns=X_train_clf.columns)

        # Remplacer les colonnes saisies par l'utilisateur
        if 'Recency'              in client.columns: client['Recency']              = recency
        if 'Frequency'            in client.columns: client['Frequency']            = frequency
        if 'MonetaryTotal'        in client.columns: client['MonetaryTotal']        = monetary
        if 'Age'                  in client.columns: client['Age']                  = age
        if 'CustomerTenureDays'   in client.columns: client['CustomerTenureDays']   = tenure

        # Normaliser
        client_scaled = scaler_clf.transform(client)

        # Prédiction Churn
        prediction  = rf.predict(client_scaled)[0]
        probabilite = rf.predict_proba(client_scaled)[0]
        prob_churn  = round(probabilite[1] * 100, 1)
        prob_fidele = round(probabilite[0] * 100, 1)
        label       = 'Churner ⚠️' if prediction == 1 else 'Fidèle ✅'

        # Prédiction Segment
        client_full_scaled = scaler.transform(
            pd.DataFrame([X_train.mean()], columns=X_train.columns)
        )
        segment_id = kmeans.predict(client_full_scaled)[0]
        labels_seg = {
            0: 'Segment A — Clients actifs',
            1: 'Segment B — Clients réguliers',
            2: 'Segment C — Clients occasionnels',
            3: 'Segment D — Clients inactifs',
        }
        segment_label = labels_seg.get(int(segment_id), f'Segment {segment_id}')

        return render_template('index.html',
            prediction   = label,
            prob_churn   = prob_churn,
            prob_fidele  = prob_fidele,
            segment      = segment_label,
            recency      = recency,
            frequency    = frequency,
            monetary     = monetary,
            age          = age,
            tenure       = tenure,
            error        = None
        )

    except Exception as e:
        return render_template('index.html',
            prediction=None,
            error=f"Erreur : {str(e)}"
        )

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)