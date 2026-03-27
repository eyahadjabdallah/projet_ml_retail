import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ============================================================
# CHARGEMENT
# ============================================================
def charger_donnees(chemin):
    df = pd.read_csv(chemin)
    print(f"✅ Chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def charger_train_test():
    X_train = pd.read_csv('data/train_test/X_train.csv')
    X_test  = pd.read_csv('data/train_test/X_test.csv')
    y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()
    y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()
    print(f"✅ Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test

def charger_modele(nom_fichier):
    chemin = f'models/{nom_fichier}'
    modele = joblib.load(chemin)
    print(f"✅ Modèle chargé : {chemin}")
    return modele

def sauvegarder_modele(modele, nom_fichier):
    os.makedirs('models', exist_ok=True)
    chemin = f'models/{nom_fichier}'
    joblib.dump(modele, chemin)
    print(f"✅ Modèle sauvegardé : {chemin}")

# ============================================================
# VISUALISATIONS
# ============================================================
def sauvegarder_figure(nom_fichier):
    os.makedirs('reports', exist_ok=True)
    chemin = f'reports/{nom_fichier}'
    plt.savefig(chemin, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Figure sauvegardée : {chemin}")

def plot_importance_features(modele, feature_names, top_n=20):
    importances = pd.Series(
        modele.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title(f'Top {top_n} features les plus importantes')
    plt.tight_layout()
    sauvegarder_figure('feature_importance.png')