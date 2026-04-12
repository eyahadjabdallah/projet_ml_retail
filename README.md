# Projet ML Retail — Analyse Comportementale Clientèle

Projet de Machine Learning réalisé dans le cadre du module GI2.
Contexte : entreprise e-commerce de cadeaux souhaitant mieux comprendre
sa clientèle, réduire le **churn** et optimiser son chiffre d'affaires.

---

## Installation

### 1. Cloner le projet
```bash
git clone https://github.com/eyahadjabdallah/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Structure du projet
```
projet_ml_retail/
├── data/
│   ├── raw/                        # Données brutes originales (CSV)
│   ├── processed/                  # Données nettoyées
│   └── train_test/                 # Données splittées (train/test)
├── notebooks/
│   └── exploration.ipynb           # Analyse exploratoire des données
├── src/
│   ├── preprocessing.py            # Nettoyage et préparation des données
│   ├── train_model.py              # Entraînement des modèles ML
│   ├── predict.py                  # Prédictions sur nouvelles données
│   └── utils.py                    # Fonctions utilitaires réutilisables
├── models/                         # Modèles sauvegardés (.pkl)
├── app/
│   ├── app.py                      # Application web Flask
│   └── templates/
│       └── index.html              # Interface utilisateur
├── reports/                        # Graphiques et visualisations
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Fichiers exclus de GitHub
└── README.md                       # Documentation du projet
```

---

## Guide d'utilisation

### Étape 1 — Préparation des données
```bash
python src/preprocessing.py
```
Ce script effectue :

- Suppression des colonnes inutiles (`NewsletterSubscribed`, `CustomerID`)
- Correction des valeurs aberrantes (`SupportTicketsCount`, `SatisfactionScore`)
- Parsing de `RegistrationDate` → `RegYear`, `RegMonth`, `RegDay`, `RegWeekday`
- Transformation de `LastLoginIP` → `IsPrivateIP` (privée/publique)
- Feature engineering avancé : `MonetaryPerDay`, `AvgBasketValue`, `TenureRatio`, `MonetaryPerFrequency`, `RecencyLog`, `FrequencyLog`, `CustomerTenureLog`, `AvgDaysBetween_RecencyRatio`
- Suppression des colonnes de leakage (liées au churn)
- Suppression de la multicolinéarité (seuil > 0.8)
- Encodage des variables catégorielles (Ordinal, One-Hot)
- Imputation médiane (fit sur train, appliquée sur test)
- Normalisation `StandardScaler` (fit sur train uniquement)
- Split stratifié 80/20

Fichiers générés :
data/processed/data_clean.csv
data/train_test/X_train.csv, X_test.csv, y_train.csv, y_test.csv
models/scaler.pkl, models/mediane_train.pkl

---

### Étape 2 — Entraînement des modèles
```bash
python src/train_model.py
```

Ce script entraîne 3 modèles de classification + 1 régression :

| Modèle | Technique | Accuracy (test) |
|--------|-----------|----------------|
| Random Forest | SMOTE (ratio 0.5), GridSearchCV, seuil optimisé | 91,2 % |
| XGBoost | SMOTE (ratio 0.5), RandomizedSearchCV, seuil fin | 97,1 % |
| Stacking | LogisticRegression sur RF + XGB | 96,9 % |
| Régression linéaire | Prédiction de `MonetaryTotal` | R² = 0,390 |

En plus :
- **ACP** : réduction de dimension (87 → 56 composantes, 95,5 % variance)
- **K-Means** : segmentation clients en 4 clusters (silhouette = 0,038)

Fichiers générés :
models/pca.pkl  models/kmeans.pkl
models/random_forest.pkl  models/xgboost_ultime.pkl  models/stacking.pkl
models/linear_regression.pkl  models/scaler_regression.pkl
reports/  (matrices de confusion, courbes ACP, importance des features)

---

### Étape 3 — Prédictions
```bash
python src/predict.py
```
Utilise les modèles sauvegardés pour prédire sur `X_test` et génère `reports/predictions_test.csv`.

---

### Étape 4 — Interface web Flask
```bash
python app/app.py
```
Ouvrir le navigateur sur : `http://127.0.0.1:5000`

Saisir les caractéristiques d'un client pour obtenir :
- Churn prédit (Fidèle / Churner)
- Probabilités associées
- Segment client (cluster K-Means)

---

## Résultats détaillés

### Classification (Churn)

| Modèle | Accuracy | Précision (Churner) | Recall (Churner) | F1-score (Churner) |
|--------|----------|---------------------|------------------|--------------------|
| Random Forest | 91,2 % | 0,88 | 0,85 | 0,86 |
| XGBoost (opt.) | 97,1 % | 0,99 | 0,92 | 0,96 |
| Stacking | 96,9 % | 0,97 | 0,93 | 0,95 |

**Pourquoi XGBoost surpasse Random Forest ?**
- Boosting itératif corrige les erreurs précédentes
- Régularisation L1/L2 intégrée
- Optimisation du seuil de décision (0,60 au lieu de 0,50) améliore le recall des churners

### Régression (MonetaryTotal)

- **RMSE** : 8 860 £
- **R²** : 0,390 *(explicable : le montant total dépend de facteurs non disponibles dans les données)*

### Clustering (K-Means)

- 4 segments clients (répartition : 1735, 1424, 337, 1 client)
- Score de silhouette moyen : 0,038 *(segments peu séparés, acceptable pour des données réelles)*

---

## Problèmes de qualité traités

| Type | Features concernées | Traitement |
|------|---------------------|------------|
| Valeurs manquantes | `Age`, `AvgDaysBetweenPurchases` | Imputation par médiane (train) |
| Valeurs aberrantes | `SupportTicketsCount` (999, -1) | Remplacement par NaN → imputation |
| Valeurs aberrantes | `SatisfactionScore` (99, -1) | Idem |
| Formats inconsistants | `RegistrationDate` (3 formats) | `pd.to_datetime(dayfirst=True)` |
| Feature inutile | `NewsletterSubscribed` (toujours Yes) | Suppression |
| Données brutes | `LastLoginIP` | Extraction `IsPrivateIP` |
| Déséquilibre classes | Churn (33 % / 67 %) | SMOTE (`sampling_strategy=0.5`) |
| Multicolinéarité | 11 paires corrélées > 0,8 | Suppression d'une colonne par paire |
| Leakage | Colonnes corrélées > 0,85 avec Churn | Suppression avant split |

---

## Améliorations apportées (version finale)

- **Feature engineering avancé** : logs, ratios, transformations non linéaires
- **Random Forest** : GridSearchCV + SMOTE + optimisation du seuil
- **XGBoost** : RandomizedSearchCV (50 itérations) + seuil fin (pas 0,002)
- **Stacking** : combinaison RF + XGB via régression logistique
- **Suppression stricte du leakage** : corrélation > 0,85 → arrêt du script
- **Visualisations** : matrices de confusion, importance des features, variance ACP

---

## Technologies utilisées

- Python 3.10
- `pandas`, `numpy`
- `scikit-learn`, `imbalanced-learn`, `xgboost`
- `matplotlib`, `seaborn`
- `Flask`
- `joblib`

---

## Auteur

**Eya Hadj Abdallah** – [GitHub](https://github.com/eyahadjabdallah)

---

## Licence

Projet pédagogique – libre d'utilisation pour l'apprentissage.