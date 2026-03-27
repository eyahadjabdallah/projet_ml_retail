# Projet ML Retail — Analyse Comportementale Clientèle

Projet de Machine Learning réalisé dans le cadre du module GI2.
Contexte : entreprise e-commerce de cadeaux souhaitant mieux comprendre
sa clientèle, réduire le churn et optimiser son chiffre d'affaires.

---

## Installation

### 1. Cloner le projet
git clone https://github.com/TON_USERNAME/projet_ml_retail.git
cd projet_ml_retail

### 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

### 3. Installer les dépendances
pip install -r requirements.txt

---

## Structure du projet

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

---

## Guide d'utilisation

### Étape 1 — Préparation des données
python src/preprocessing.py

Ce script effectue dans l'ordre :
- Suppression des features inutiles (NewsletterSubscribed, CustomerID)
- Correction des valeurs aberrantes (999, -1, 99)
- Imputation des valeurs manquantes (médiane)
- Parsing de RegistrationDate en RegYear, RegMonth, RegDay
- Transformation de LastLoginIP en IsPrivateIP
- Feature Engineering (MonetaryPerDay, AvgBasketValue, TenureRatio)
- Suppression de la multicolinéarité (seuil > 0.8)
- Encodage des features catégorielles (Ordinal, One-Hot, Target)
- Normalisation StandardScaler + Split 80/20 stratifié

Fichiers générés :
- data/processed/data_clean.csv
- data/train_test/X_train.csv
- data/train_test/X_test.csv
- data/train_test/y_train.csv
- data/train_test/y_test.csv
- models/scaler.pkl

### Étape 2 — Entraînement des modèles
python src/train_model.py

Ce script entraîne 3 modèles :
- ACP : réduction de 75 features → 49 composantes (95% variance)
- K-Means : segmentation clients en 4 clusters
- Random Forest : prédiction Churn avec SMOTE + GridSearchCV
- Régression Linéaire : prédiction MonetaryTotal

Fichiers générés :
- models/pca.pkl
- models/kmeans.pkl
- models/random_forest.pkl
- models/linear_regression.pkl
- reports/*.png

### Étape 3 — Prédictions
python src/predict.py

Prédit sur X_test et affiche les résultats.
Fichier généré : reports/predictions_test.csv

### Étape 4 — Interface web Flask
python app/app.py

Ouvrir le navigateur sur : http://127.0.0.1:5000
Saisir les informations d'un client et obtenir :
- Prédiction Churn (Fidèle ou Churner)
- Probabilités associées
- Segment client (A, B, C ou D)

---

## Modèles et résultats

| Modèle            | Tâche                     | Résultat                        |
|-------------------|---------------------------|---------------------------------|
| K-Means           | Clustering clients        | Silhouette = 0.078, k=4         |
| Random Forest     | Prédiction Churn          | Optimisé via GridSearchCV + SMOTE |
| Régression linéaire | Prédiction MonetaryTotal | R² = 0.395, RMSE = 8824 £      |
| ACP               | Réduction de dimension    | 75 → 49 composantes (95%)       |

---

## Problèmes de qualité traités

- Valeurs manquantes : Age (30%), AvgDaysBetweenPurchases (1.8%)
- Valeurs aberrantes : SupportTicketsCount (999, -1), SatisfactionScore (99, -1)
- Formats inconsistants : RegistrationDate (3 formats différents)
- Feature inutile : NewsletterSubscribed (variance nulle)
- Données brutes : LastLoginIP transformé en IsPrivateIP
- Déséquilibre classes : Churn 33%/67% traité avec SMOTE
- Multicolinéarité : 11 colonnes supprimées (seuil > 0.8)

---

## Technologies utilisées

- Python 3.10
- pandas, numpy
- scikit-learn, imbalanced-learn
- matplotlib, seaborn
- Flask
- joblib

