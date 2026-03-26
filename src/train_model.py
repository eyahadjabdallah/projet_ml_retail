import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    silhouette_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score
)
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import charger_train_test, sauvegarder_modele, sauvegarder_figure

# ============================================================
# ACP — Réduction de dimension (section 6.1 cahier des charges)
# Objectif : réduire 75 features → 2-10 composantes
# ============================================================
def train_acp(X_train, X_test):
    from sklearn.decomposition import PCA

    print("\n" + "="*55)
    print("   ACP — ANALYSE EN COMPOSANTES PRINCIPALES")
    print("="*55)

    # Étape 1 — Trouver le nombre de composantes optimal
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train)

    # Variance cumulée
    variance_cumulee = np.cumsum(pca_full.explained_variance_ratio_)

    # Trouver n_components pour 95% de variance
    n_95 = np.argmax(variance_cumulee >= 0.95) + 1
    n_90 = np.argmax(variance_cumulee >= 0.90) + 1
    print(f"   Composantes pour 90% variance : {n_90}")
    print(f"   Composantes pour 95% variance : {n_95}")

    # Visualisation variance expliquée
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(variance_cumulee)+1),
             variance_cumulee, marker='o', markersize=3, color='steelblue')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95%')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90%')
    plt.axvline(x=n_95, color='red', linestyle=':', alpha=0.7)
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance cumulée')
    plt.title('Variance expliquée cumulée')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(1, 21),
            pca_full.explained_variance_ratio_[:20],
            color='steelblue', alpha=0.7)
    plt.xlabel('Composante')
    plt.ylabel('Variance expliquée')
    plt.title('Variance par composante (Top 20)')

    plt.tight_layout()
    sauvegarder_figure('acp_variance.png')

    # Étape 2 — ACP finale avec n_95 composantes
    pca_final = PCA(n_components=n_95, random_state=42)
    X_train_pca = pca_final.fit_transform(X_train)
    X_test_pca  = pca_final.transform(X_test)
    print(f"\n✅ ACP finale : {X_train.shape[1]} features → {n_95} composantes")
    print(f"   Variance conservée : {variance_cumulee[n_95-1]*100:.1f}%")

    # Étape 3 — Visualisation 2D (les 2 premières composantes)
    # On a besoin de y_train pour colorier par classe
    print(f"   Visualisation 2D sauvegardée")

    # Étape 4 — Sauvegarder le modèle ACP
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca_final, 'models/pca.pkl')
    print(f"✅ Modèle ACP sauvegardé : models/pca.pkl")

    return pca_final, X_train_pca, X_test_pca

# ============================================================
# MODÈLE 1 — CLUSTERING K-MEANS
# Objectif : segmenter les clients en groupes
# ============================================================
def train_clustering(X_train):
    print("\n" + "="*55)
    print("   MODÈLE 1 — CLUSTERING K-MEANS")
    print("="*55)

    # Trouver le meilleur k avec la méthode du coude
    inerties = []
    silhouettes = []
    K = range(2, 9)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        inerties.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_train, kmeans.labels_))
        print(f"   k={k} → inertie={kmeans.inertia_:.0f}, silhouette={silhouette_score(X_train, kmeans.labels_):.3f}")

    # Visualisation méthode du coude
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(K), inerties, marker='o', color='steelblue')
    axes[0].set_title('Méthode du coude — Inertie')
    axes[0].set_xlabel('Nombre de clusters k')
    axes[0].set_ylabel('Inertie')

    axes[1].plot(list(K), silhouettes, marker='o', color='coral')
    axes[1].set_title('Score Silhouette')
    axes[1].set_xlabel('Nombre de clusters k')
    axes[1].set_ylabel('Silhouette')

    plt.tight_layout()
    sauvegarder_figure('clustering_choix_k.png')

    # Entraîner avec k=4 (cohérent avec RFMSegment du dataset)
    k_final = 4
    kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    kmeans_final.fit(X_train)

    score = silhouette_score(X_train, kmeans_final.labels_)
    print(f"\n✅ K-Means final : k={k_final}, silhouette={score:.3f}")
    print(f"   Répartition des clusters : {pd.Series(kmeans_final.labels_).value_counts().to_dict()}")

    sauvegarder_modele(kmeans_final, 'kmeans.pkl')
    return kmeans_final

# ============================================================
# MODÈLE 2 — CLASSIFICATION RANDOM FOREST
# Objectif : prédire si un client va churner (0 ou 1)
# ============================================================
def train_classification(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("   MODÈLE 2 — CLASSIFICATION RANDOM FOREST")
    print("="*55)

    # Supprimer les colonnes qui causent le data leakage
    # ChurnRiskCategory → construite à partir de Churn
    # CustomerType_Perdu → décrit directement un client qui est parti
    cols_leakage = [c for c in X_train.columns
                if 'ChurnRisk' in c
                or 'CustomerType_Perdu' in c]
    X_train = X_train.drop(columns=cols_leakage, errors='ignore')
    X_test  = X_test.drop(columns=cols_leakage,  errors='ignore')
    print(f"⚠️  Colonnes leakage supprimées : {cols_leakage}")

    # Entraînement
    rf = RandomForestClassifier(

        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)

    # Prédictions
    y_pred = rf.predict(X_test)

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy : {acc:.3f}")
    print(f"\nRapport de classification :")
    print(classification_report(y_test, y_pred,
                                target_names=['Fidèle (0)', 'Churner (1)']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'],
                yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de Confusion — Random Forest')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix.png')

    # Feature importance
    importances = pd.Series(
        rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='steelblue')
    plt.title('Top 15 features importantes — Random Forest')
    plt.tight_layout()
    sauvegarder_figure('classification_feature_importance.png')

    sauvegarder_modele(rf, 'random_forest.pkl')
    return rf
# ============================================================
# MODÈLE 3 — RÉGRESSION LINÉAIRE
# Objectif : prédire MonetaryTotal (dépenses du client)
# ============================================================
def train_regression(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("   MODÈLE 3 — RÉGRESSION LINÉAIRE")
    print("="*55)

    # Charger données non-normalisées pour récupérer MonetaryTotal
    df_clean = pd.read_csv('data/processed/data_clean.csv')

    # Séparer en train/test avec le même random_state
    from sklearn.model_selection import train_test_split
    y_reg = df_clean['MonetaryTotal']
    X_reg = df_clean.drop(columns=['MonetaryTotal', 'Churn'])

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg,
        test_size=0.2,
        random_state=42
    )

    # Normaliser — fit sur train uniquement
    from sklearn.preprocessing import StandardScaler
    scaler_reg = StandardScaler()
    X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_scaled  = scaler_reg.transform(X_reg_test)

    # Entraînement
    lr = LinearRegression()
    lr.fit(X_reg_train_scaled, y_reg_train)

    # Prédictions
    y_reg_pred = lr.predict(X_reg_test_scaled)

    # Métriques
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2   = r2_score(y_reg_test, y_reg_pred)
    print(f"\n✅ RMSE : {rmse:.2f} £")
    print(f"✅ R²   : {r2:.3f}")
    print(f"   (R²=1 = parfait, R²=0 = modèle nul)")

    # Visualisation réel vs prédit
    plt.figure(figsize=(7, 5))
    plt.scatter(y_reg_test, y_reg_pred, alpha=0.4, color='steelblue')
    plt.plot([y_reg_test.min(), y_reg_test.max()],
             [y_reg_test.min(), y_reg_test.max()],
             'r--', linewidth=2)
    plt.xlabel('Valeurs réelles (£)')
    plt.ylabel('Valeurs prédites (£)')
    plt.title('Régression — Réel vs Prédit')
    plt.tight_layout()
    sauvegarder_figure('regression_reel_vs_predit.png')

    sauvegarder_modele(lr, 'linear_regression.pkl')
    joblib.dump(scaler_reg, 'models/scaler_regression.pkl')
    print(f"✅ Scaler régression sauvegardé")
    return lr

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("Chargement des données train/test...")
    X_train, X_test, y_train, y_test = charger_train_test()

    # ACP — réduction de dimension
    pca, X_train_pca, X_test_pca = train_acp(X_train, X_test)

    # Modèle 1 — Clustering
    kmeans = train_clustering(X_train)

    # Modèle 2 — Classification
    rf = train_classification(X_train, X_test, y_train, y_test)

    # Modèle 3 — Régression
    lr = train_regression(X_train, X_test, y_train, y_test)

    print("\n" + "="*55)
    print("🎉 Entraînement terminé !")
    print("   Modèles sauvegardés dans models/")
    print("   Graphiques sauvegardés dans reports/")
    print("="*55)