# Projet ML Retail — Analyse Comportementale Clientèle

**Atelier Machine Learning — Module GI2 · Année universitaire 2025-2026**
Encadrante : Mme Fadoua Drira · Réalisé par : Eya Hadj Abdallah

---

## Description du projet

Projet de Machine Learning réalisé dans le cadre du module GI2.
Contexte : une entreprise e-commerce de cadeaux souhaite mieux comprendre sa clientèle, réduire le taux de churn (départ des clients) et optimiser son chiffre d'affaires à partir d'une base de données de 4 372 clients décrite par 52 variables.

La base est volontairement imparfaite (valeurs manquantes, outliers, data leakage) afin de maîtriser l'ensemble de la chaîne de traitement en data science :

```
Exploration → Préparation → Modélisation → Évaluation → Déploiement
```

---

## Résultats clés

### Classification — Prédiction du Churn

| Modèle | Accuracy | F1-score (Churner) | Rappel | AUC | Seuil |
|---|---|---|---|---|---|
| Random Forest | 94.5% | 0.92 | 0.89 | 0.985 | 0.47 |
| XGBoost | 97.1% | 0.96 | 0.92 | — | 0.60 |
| Stacking (RF + XGB) | 97.0% | 0.95 | 0.93 | — | 0.50 |

### Autres tâches

| Tâche | Modèle | Résultat |
|---|---|---|
| Régression (MonetaryTotal) | XGBoost Regressor | R² = 0.545 · RMSE = 7 648 £ |
| Clustering | K-Means (k=4, sur ACP) | Silhouette = 0.086 |
| Réduction dimensionnelle | ACP | 95.5% variance avec 56 composantes |

---

## Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/                        # Données brutes originales (CSV)
│   ├── processed/                  # Données nettoyées (data_clean.csv)
│   └── train_test/                 # X_train, X_test, y_train, y_test
│
├── notebooks/
│   └── exploration.ipynb           # Analyse exploratoire (EDA)
│
├── src/
│   ├── preprocessing.py            # Pipeline complet de préparation
│   ├── train_model.py              # Entraînement de tous les modèles
│   ├── predict.py                  # Prédictions sur nouvelles données
│   └── utils.py                    # Fonctions utilitaires partagées
│
├── models/                         # Modèles sérialisés (.pkl)
│   ├── scaler.pkl
│   ├── mediane_train.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── stacking.pkl
│   ├── regression_xgboost_optimized.pkl
│   └── scaler_regression.pkl
│
├── app/
│   ├── app.py                      # Application web Flask
│   ├── static/
│   │   └── script.js               # JavaScript du dashboard
│   └── templates/
│       └── index.html              # Interface utilisateur
│
├── reports/                        # Graphiques et visualisations (.png)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/eyahadjabdallah/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Exécution du pipeline

### Etape 1 — Préparation des données

```bash
python src/preprocessing.py
```

Ce script effectue dans l'ordre :

1. Suppression des colonnes inutiles (NewsletterSubscribed, CustomerID)
2. Correction des valeurs aberrantes (SupportTicketsCount : 999, -1 → NaN ; SatisfactionScore : 99, -1 → NaN)
3. Parsing de RegistrationDate (3 formats UK/ISO/US) → RegYear, RegMonth, RegDay, RegWeekday
4. Transformation de LastLoginIP → booléen IsPrivateIP
5. Feature engineering : MonetaryPerDay, AvgBasketValue, TenureRatio
6. Suppression du data leakage (colonnes corrélées à Churn au-delà de 0.85)
7. Suppression de la multicolinéarité (seuil r > 0.8)
8. Encodage des variables catégorielles (Ordinal + One-Hot + Country après split)
9. Split stratifié 80/20 (random_state=42, stratify=y)
10. Imputation par la médiane (fit sur train uniquement, appliquée sur test)
11. Standardisation StandardScaler (fit sur train, transform sur test)

Fichiers générés :
- data/processed/data_clean.csv
- data/train_test/X_train.csv, X_test.csv, y_train.csv, y_test.csv
- models/scaler.pkl, models/mediane_train.pkl

### Etape 2 — Entraînement des modèles

```bash
python src/train_model.py
```

Ce script entraîne successivement :

- ACP : réduction vers 56 composantes (95.5% de variance conservée)
- K-Means : segmentation en 4 clusters sur les composantes ACP
- Random Forest : SMOTE + RandomizedSearchCV (60 combinaisons, 5 plis) + optimisation du seuil → 94.5% d'accuracy
- XGBoost : SMOTE + GridSearchCV (5 plis) + optimisation du seuil → 97.1% d'accuracy
- Stacking (RF + XGB) : meta-modèle LogisticRegression → 97.0% d'accuracy
- XGBoost Regressor : RandomizedSearchCV pour prédire MonetaryTotal → R² = 0.545

Fichiers générés dans models/ : pca.pkl, kmeans.pkl, random_forest.pkl, xgboost.pkl, stacking.pkl, regression_xgboost_optimized.pkl, scaler_regression.pkl

Graphiques générés dans reports/ : matrices de confusion, courbes ACP, importances des features, courbes ROC-AUC, réel vs prédit

### Etape 3 — Prédictions sur l'ensemble de test

```bash
python src/predict.py
```

Charge les modèles sauvegardés, génère les prédictions sur X_test et sauvegarde reports/predictions_test.csv.

### Etape 4 — Lancer l'application web Flask

```bash
python app/app.py
```

Ouvrir dans le navigateur : http://127.0.0.1:5000

L'interface permet de saisir 4 caractéristiques client et retourne :

| Sortie | Détail |
|---|---|
| Prédiction du churn | Fidèle / Churner pour RF, XGBoost et Stacking avec probabilités |
| Segment K-Means | Cluster A / B / C / D avec description du profil |
| Valeur estimée | Montant total prédit par XGBoost Regressor |
| Dashboard métriques | Accuracy, F1, Rappel, ROC-AUC en temps réel |
| Features importantes | Top 10 features (graphique interactif Chart.js) |

---

## Problèmes de qualité traités

| Type | Features concernées | Traitement |
|---|---|---|
| Valeurs manquantes | Age (30%), AvgDaysBetweenPurchases | Imputation médiane (fit sur train) |
| Valeurs aberrantes | SupportTicketsCount (999, -1), SatisfactionScore (99, -1) | Remplacement par NaN puis imputation |
| Formats inconsistants | RegistrationDate (3 formats) | pd.to_datetime(dayfirst=True) |
| Feature sans variance | NewsletterSubscribed (toujours "Yes") | Suppression |
| Données brutes | LastLoginIP | Transformation en IsPrivateIP |
| Déséquilibre des classes | Churn (33% / 67%) | SMOTE (sampling_strategy=0.5) |
| Multicolinéarité | 11 paires (ex. MonetaryMin ↔ MonetaryStd : r=0.97) | Suppression (seuil r > 0.8) |
| Data leakage | ChurnRiskCategory, CustomerType, LoyaltyLevel, RFMSegment, AccountStatus, Recency (r=0.86) | Suppression + vérification automatique |

Point critique : avant correction du data leakage, le Random Forest atteignait une accuracy de 100% (score artificiel). Après correction : 94.5% (score réaliste et généralisable).

---

## Analyse des résultats

### Pourquoi XGBoost surpasse Random Forest

1. Boosting itératif : chaque arbre corrige les erreurs du précédent.
2. Régularisation L1/L2 intégrée réduit le sur-apprentissage.
3. Optimisation du seuil à 0.60 maximise la précision sur les churners.

### Pourquoi le Stacking n'apporte pas de gain significatif

Le meta-modèle (régression logistique) surpondère naturellement XGBoost (coefficient 5.82 vs 2.52 pour RF), rendant le stacking quasi équivalent à XGBoost seul.

### Clustering — Segmentation K-Means (k=4)

| Segment | Profil |
|---|---|
| A — Clients Premium | Haute fréquence, montant élevé, faible récence |
| B — Clients Réguliers | Fréquence et montant modérés, stables |
| C — Clients Occasionnels | Fréquence faible, fort potentiel de réactivation |
| D — Clients Inactifs | Très longue récence, risque élevé de churn |

Score de silhouette : 0.086 (vs 0.038 sans ACP, soit +127% d'amélioration)

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Langage | Python 3.10 |
| Manipulation des données | pandas, numpy |
| Machine Learning | scikit-learn, xgboost, imbalanced-learn |
| Visualisation | matplotlib, seaborn |
| Déploiement | Flask, Chart.js |
| Sérialisation | joblib |

---

## Auteure

Eya Hadj Abdallah — https://github.com/eyahadjabdallah

Module Machine Learning · GI2 · 2025-2026
Encadrante : Mme Fadoua Drira

---

*Projet pédagogique — libre d'utilisation pour l'apprentissage.*