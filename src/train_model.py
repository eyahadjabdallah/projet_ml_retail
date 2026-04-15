import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')#Les bibliothèques habituelles. warnings.filterwarnings('ignore') → supprime les messages d'avertissement non critiques pour garder un affichage propre.
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    silhouette_score, classification_report,
    confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, f1_score
)
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import charger_train_test, sauvegarder_modele, sauvegarder_figure

# ============================================================
# ROC - AUC UTILITY
# ============================================================
def plot_roc_auc(y_test, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()

    sauvegarder_figure(f"roc_auc_{model_name.lower()}.png")
    print(f"✅ AUC {model_name} : {roc_auc:.3f}")

# ============================================================
# ACP
# ============================================================
def train_acp(X_train, X_test):
    print("\n" + "="*55)
    print("   ACP — ANALYSE EN COMPOSANTES PRINCIPALES")
    print("="*55)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train)
    # on cree un objet PCA sans preciser le nombre de composantes et on lentraine sur les donnees d entrainement uniquement 
    #l.ACP apprend les directions principales a partir du train (pas de fuite)
    var_cum = np.cumsum(pca_full.explained_variance_ratio_)
    #explained_variance_ratio_ est un tableau qui donne le pourcentage de variance expliqué par chaque composante.
    #np.cumsum fait la somme cumulée.
    #var_cum >= 0.95 donne un tableau de True/False : True là où la somme cumulée atteint ou dépasse 95%.
    n_95 = np.argmax(var_cum >= 0.95) + 1
    n_90 = np.argmax(var_cum >= 0.90) + 1
    #np.argmax retourne l’index du premier True. Par exemple si la somme cumulée devient >=0.95 à l’index 55 (le 56ème élément), alors argmax vaut 55.
    print(f"   Composantes pour 90% : {n_90}, pour 95% : {n_95}")
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1,len(var_cum)+1), var_cum, marker='o', markersize=3)
    plt.axhline(0.95, color='r', linestyle='--')
    plt.axvline(n_95, color='r', linestyle=':')
    plt.title('Variance cumulée')
    plt.subplot(1,2,2)
    plt.bar(range(1,21), pca_full.explained_variance_ratio_[:20])
    plt.title('Variance par composante (top 20)')
    plt.tight_layout()
    sauvegarder_figure('acp_variance.png')
    
    pca = PCA(n_components=n_95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"✅ ACP : {X_train.shape[1]} → {n_95} composantes (var={var_cum[n_95-1]*100:.1f}%)")
    joblib.dump(pca, 'models/pca.pkl')
    return pca, X_train_pca, X_test_pca

# ============================================================
# CLUSTERING K-MEANS # apprentissage non supervisee sert a segmenter la colonne churn
# ============================================================
def train_clustering(data):
    print("\n" + "="*55)
    print("   MODÈLE 1 — CLUSTERING K-MEANS")
    print("="*55)
    inerties, silhouettes = [], []
    K = range(2,9)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(data)                     # ← ici c'est data, pas X_train
        inerties.append(km.inertia_)# L’axe des y du graphique de l’inertie (courbe bleue dans la figure clustering_choix_k.png) 
        #représente la valeur de l’inertie intra-cluster (notée km.inertia_).
        #Cette valeur est la somme des carrés des distances entre chaque client et le centre de son groupe.
        silhouettes.append(silhouette_score(data, km.labels_))
        print(f"   k={k} → inertie={km.inertia_:.0f}, silhouette={silhouettes[-1]:.3f}")
    
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(K, inerties, marker='o')
    axes[0].set_title('Inertie')
    axes[1].plot(K, silhouettes, marker='o', color='coral')
    axes[1].set_title('Silhouette')
    plt.tight_layout()
    sauvegarder_figure('clustering_choix_k.png')
    
    k_final = 4
    km_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    km_final.fit(data)
    score = silhouette_score(data, km_final.labels_)
    print(f"\n✅ K-Means final : k={k_final}, silhouette={score:.3f}")
    print(f"   Répartition : {pd.Series(km_final.labels_).value_counts().to_dict()}")
    sauvegarder_modele(km_final, 'kmeans.pkl')
    return km_final
# ============================================================
# RANDOM FOREST AVEC SMOTE ET OPTIMISATION DU SEUIL
# ============================================================
def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("   MODÈLE 2a — RANDOM FOREST (SMOTE + SEUIL)")
    print("="*55)
    
    # Vérification des corrélations
    corr_check = X_train.copy()
    corr_check['Churn'] = y_train
    corr_abs = corr_check.corr()['Churn'].abs().sort_values(ascending=False)
    top_corr = corr_abs[corr_abs.index != 'Churn'].head(10)

    print("\n🔍 Top 10 corrélations avec Churn :")
    print(top_corr.round(4))

    if (top_corr > 0.85).any():
        print("\n⚠️ Leakage détecté (>0.85)")
        sys.exit(1)

    print(f"\nDistribution : {y_train.value_counts().to_dict()}")
    
    
    # Grille d'hyperparamètres (basée sur vos meilleurs résultats précédents)
    param_dist = {
        'n_estimators': [600, 800, 1000, 1200],#nombre d'arbres dans la foret
        'max_depth': [20, 30, 40, None],#profondeur maximale de chaque arbre.Limiter la profondeurevite le surapprentissage
        'min_samples_split': [2, 3, 5],#nombre minimum d’échantillons requis pour diviser un nœud.
        'min_samples_leaf': [1, 2, 3],#nombre minimum d’échantillons dans une feuille.
        'max_features': ['sqrt', 'log2', 0.7],
        'bootstrap': [True],
        'criterion': ['gini'],
        'max_samples': [0.8, 0.9, None]
    }

    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    #random_state=42 : fixe la graine aléatoire pour l’initialisation des arbres et le tirage des sous-ensembles. Permet la reproductibilité.
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=60,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    #Lance la recherche. Pour chaque combinaison d’hyperparamètres, on fait une validation croisée 5 plis,
    # on calcule l’accuracy moyenne, et on garde la meilleure combinaison.
    
    best_rf = search.best_estimator_
    print(f"✅ Meilleurs hyperparamètres : {search.best_params_}")
    
    # Optimisation du seuil
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train#préserve la proportion de churners dans les deux sous-ensembles.
    )#on va chercher le meilleur seuil on prend les donnees deja d entrainement deja equilibree par SMOTE(X_train_bal ,y_train_bal)
    #on les divise a nouveau:80% pour sous-entrainement,20% pour validation et cette c est pour tester le seuils pas pour evaluer le modle final
    best_rf.fit(X_train_sub, y_train_sub)
    val_proba = best_rf.predict_proba(X_val)[:, 1]#on recupere les probabilites de churn pour les clients de validation
    
    thresholds = np.arange(0.25, 0.75, 0.01)# on teste tous les seuils de 0.3 a 0.7 par pas de 0.01
    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\n✅ Meilleur seuil : {best_thresh:.2f} | F1={best_f1:.3f}")
    
    best_rf.fit(X_train, y_train)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_opt = (y_proba >= best_thresh).astype(int)

    acc_default = accuracy_score(y_test, y_pred_default)
    acc_opt = accuracy_score(y_test, y_pred_opt)

    print(f"\nAccuracy (0.5)   : {acc_default:.3f}")
    print(f"Accuracy (opt)    : {acc_opt:.3f}")

    y_pred = y_pred_opt if acc_opt > acc_default else y_pred_default

    print("\n📊 Classification Report :")
    print(classification_report(y_test, y_pred))
    

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'], yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de confusion - Random Forest')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix_rf.png')
    
    # Importance des features
    importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10,6))
    importances.plot(kind='bar', color='steelblue')
    plt.title('Top 15 features importantes - Random Forest')
    plt.tight_layout()
    sauvegarder_figure('classification_feature_importance_rf.png')
    
    sauvegarder_modele(best_rf, 'random_forest.pkl')
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    plot_roc_auc(y_test, y_proba_rf, "RandomForest")
    return best_rf

# ============================================================
# XGBOOST AVEC OPTIMISATION DU SEUIL
# ============================================================
def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("   MODÈLE 2b — XGBOOST (OPTIMISÉ + SEUIL)")
    print("="*55)
    
    print(f"\n   Distribution initiale : {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"   Après SMOTE (ratio 0.5) : {pd.Series(y_train_sm).value_counts().to_dict()}")
    
    param_grid = {
        'n_estimators': [500, 700],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.03, 0.05, 0.08],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train_sm, y_train_sm)
    
    print(f"✅ Meilleurs hyperparamètres : {grid.best_params_}")
    best_xgb = grid.best_estimator_
    
    # Optimisation du seuil
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_sm, y_train_sm, test_size=0.2, random_state=42, stratify=y_train_sm
    )
    best_xgb.fit(X_train_sub, y_train_sub)
    val_proba = best_xgb.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_thresh = 0.5
    best_acc = 0
    for thresh in thresholds:
        pred = (val_proba >= thresh).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    print(f"✅ Meilleur seuil : {best_thresh:.2f} (accuracy validation {best_acc:.3f})")
    
    # Ré-entraînement final
    best_xgb.fit(X_train_sm, y_train_sm)
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy test (seuil optimisé) : {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Fidèle (0)', 'Churner (1)']))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'], yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de confusion - XGBoost')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix_xgb.png')
    
    importances = pd.Series(best_xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10,6))
    importances.plot(kind='bar', color='steelblue')
    plt.title('Top 15 features importantes - XGBoost')
    plt.tight_layout()
    sauvegarder_figure('classification_feature_importance_xgb.png')
    
    sauvegarder_modele(best_xgb, 'xgboost.pkl')
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    plot_roc_auc(y_test, y_proba_xgb, "XGBoost")
    return best_xgb

# ============================================================
# STACKING (RF + XGB)
# ============================================================
def train_stacking(X_train, X_test, y_train, y_test, rf_model, xgb_model):
    print("\n" + "="*55)
    print("   MODÈLE 2c — STACKING (RF + XGB)")
    print("="*55)
    
    rf_pred_train = cross_val_predict(rf_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    xgb_pred_train = cross_val_predict(xgb_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    rf_pred_test = rf_model.predict_proba(X_test)[:, 1]
    xgb_pred_test = xgb_model.predict_proba(X_test)[:, 1]
    
    X_meta_train = np.column_stack((rf_pred_train, xgb_pred_train))
    X_meta_test = np.column_stack((rf_pred_test, xgb_pred_test))
    
    meta = LogisticRegression(random_state=42)
    meta.fit(X_meta_train, y_train)
    y_pred_meta = meta.predict(X_meta_test)
    acc_meta = accuracy_score(y_test, y_pred_meta)
    print(f"\n✅ Accuracy test (stacking) : {acc_meta:.3f}")
    print(classification_report(y_test, y_pred_meta, target_names=['Fidèle (0)', 'Churner (1)']))
    
    cm = confusion_matrix(y_test, y_pred_meta)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churner'], yticklabels=['Fidèle', 'Churner'])
    plt.title('Matrice de confusion - Stacking')
    plt.tight_layout()
    sauvegarder_figure('classification_confusion_matrix_stacking.png')
    
    coefs = pd.Series(meta.coef_[0], index=['RF', 'XGB'])
    print("\nPoids des modèles dans le stacking :")
    print(coefs)
    
    sauvegarder_modele(meta, 'stacking.pkl')
    y_proba_stack = meta.predict_proba(X_meta_test)[:, 1]
    plot_roc_auc(y_test, y_proba_stack, "Stacking")
    return meta

# ============================================================
# RÉGRESSION NON LINEAIRE
# ============================================================
def train_regression():
    print("\n" + "="*55)
    print("   MODÈLE 3 — RÉGRESSION (XGBoost OPTIMISÉ)")
    print("="*55)
    
    df = pd.read_csv('data/processed/data_clean.csv')
    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])
    
    X = df.drop(columns=['MonetaryTotal', 'Churn'])
    y = df['MonetaryTotal']
    
    # Nettoyage de la cible
    y = y.replace([np.inf, -np.inf], np.nan)
    if y.isnull().any():
        y = y.fillna(y.median())
    
    # Détection valeurs négatives
    if (y < 0).any():
        print("⚠️ Valeurs négatives détectées → pas de transformation logarithmique")
        y_transformed = y
        use_log = False
    else:
        y_transformed = np.log1p(y)
        use_log = True
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
    
    mediane = X_train.median()
    X_train = X_train.fillna(mediane)
    X_test = X_test.fillna(mediane)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV
    
    # Grille d'hyperparamètres
    param_dist = {
        'n_estimators': [300, 500, 700],
        'max_depth': [5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.07],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    xgb = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(xgb, param_dist, n_iter=30, cv=5, 
                                       scoring='r2', n_jobs=-1, random_state=42, verbose=1)
    random_search.fit(X_train_scaled, y_train)
    
    print(f"✅ Meilleurs hyperparamètres : {random_search.best_params_}")
    best_model = random_search.best_estimator_
    
    y_pred_transformed = best_model.predict(X_test_scaled)
    
    if use_log:
        y_pred = np.expm1(y_pred_transformed)
        y_test_orig = np.expm1(y_test)
    else:
        y_pred = y_pred_transformed
        y_test_orig = y_test
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    print(f"\n✅ RMSE : {rmse:.2f} £")
    print(f"✅ R²   : {r2:.3f}")
    
    plt.figure(figsize=(7,5))
    plt.scatter(y_test_orig, y_pred, alpha=0.4, color='steelblue')
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('Valeurs réelles (£)')
    plt.ylabel('Valeurs prédites (£)')
    plt.title('Régression XGBoost optimisée - Réel vs Prédit')
    plt.tight_layout()
    sauvegarder_figure('regression_reel_vs_predit.png')
    
    sauvegarder_modele(best_model, 'regression_xgboost_optimized.pkl')
    joblib.dump(scaler, 'models/scaler_regression.pkl')
    return best_model
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Chargement des données train/test pour classification...")
    X_train, X_test, y_train, y_test = charger_train_test()
    
    pca, X_train_pca, X_test_pca = train_acp(X_train, X_test)
    kmeans = train_clustering(X_train_pca)
    
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)
    stacking_model = train_stacking(X_train, X_test, y_train, y_test, rf_model, xgb_model)
    
    lr_model = train_regression()
    
    print("\n" + "="*55)
    print("🎉 Entraînement terminé !")
    print("   Modèles sauvegardés dans models/")
    print("   Graphiques dans reports/")
    print("="*55)