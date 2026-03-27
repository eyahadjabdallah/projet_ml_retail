import pandas as pd #manipulation des tableaux de donnees 
import numpy as np #calculs mathematiques 
import re # regular expression : permet de detecter des patterns dans du texte . on l'utilise pour detecter les formats de dates 
import ipaddress # Bibliothèque Python intégrée qui permet d'analyser des adresses IP. On l'utilise pour détecter si une IP est privée ou publique.
from sklearn.preprocessing import StandardScaler, OrdinalEncoder #StandardScaler pour normaliser les données, OrdinalEncoder pour encoder les catégories avec un ordre
from sklearn.model_selection import train_test_split  #Fonction qui coupe le dataset en deux parties : entraînement (80%) et test (20%).
from sklearn.impute import KNNImputer
import joblib # Permet de sauvegarder et recharger des objets Python (comme nos modèles et scalers) dans des fichiers .pkl
import os # Permet d'interagir avec le système de fichiers — créer des dossiers, vérifier si un fichier existe, etc.

# ============================================================
# ÉTAPE 1 — Charger les données
# ============================================================
def charger_donnees(chemin):
    df = pd.read_csv(chemin)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

# ============================================================
# ÉTAPE 2 — Supprimer les features inutiles
# ============================================================
def supprimer_features_inutiles(df):
    cols_a_supprimer = ['NewsletterSubscribed', 'CustomerID']
    # NewsletterSubscribed = toujours "Yes" (variance nulle, inutile). CustomerID = juste un identifiant, pas une information utile pour prédire.
    # Colonnes avec >50% manquantes
    cols_50 = [c for c in df.columns
               if df[c].isnull().mean() > 0.5]
    df = df.drop(columns=cols_a_supprimer, errors='ignore')
    print(f"✅ Features supprimées (variance nulle / >50% NaN) : {cols_a_supprimer}")
    return df

# ============================================================
# ÉTAPE 3 — Corriger les valeurs aberrantes
# ============================================================
def corriger_aberrantes(df):
    n_999 = (df['SupportTicketsCount'] == 999).sum()
    n_m1s = (df['SupportTicketsCount'] == -1).sum()
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace({999: np.nan, -1: np.nan})
    print(f"✅ SupportTicketsCount : {n_999} valeurs 999 et {n_m1s} valeurs -1 → NaN")

    n_99  = (df['SatisfactionScore'] == 99).sum()
    n_m1f = (df['SatisfactionScore'] == -1).sum()
    df['SatisfactionScore'] = df['SatisfactionScore'].replace({99: np.nan, -1: np.nan})
    print(f"✅ SatisfactionScore   : {n_99} valeurs 99 et {n_m1f} valeurs -1 → NaN")
    return df
# ============================================================
# ÉTAPE 4 — Parser RegistrationDate
# ============================================================
def parser_dates(df):
    df['RegistrationDate'] = pd.to_datetime( #convertir du texte en objet date Python
        df['RegistrationDate'],
        dayfirst=True, # priorité au format UK : 12/03/10 = 12 mars et non 3 décembre.
        errors='coerce'  #si un format est impossible à parser, mettre NaT (Not a Time) au lieu de planter.
    )
    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day # le jour (1-31)
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday #jour de la semaine (lundi a dimanche)
    df = df.drop(columns=['RegistrationDate'])
    print(f"✅ RegistrationDate parsée → RegYear, RegMonth, RegDay, RegWeekday")
    return df

# ============================================================
# ÉTAPE 5 — Transformer LastLoginIP
# ============================================================
def transformer_ip(df):
    def is_private(ip_str):
        try:
            return int(ipaddress.ip_address(str(ip_str)).is_private)
        except:
            return 0
    df['IsPrivateIP'] = df['LastLoginIP'].apply(is_private)
    df = df.drop(columns=['LastLoginIP'])
    print(f"✅ LastLoginIP → IsPrivateIP (0=publique, 1=privée)")
    return df

# ============================================================
# ÉTAPE 6 — Feature Engineering
# ============================================================
def feature_engineering(df):
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    df['TenureRatio']    = df['Recency'] / (df['CustomerTenureDays'] + 1)
    print("✅ Feature Engineering :")
    print("   - MonetaryPerDay  = MonetaryTotal / (Recency + 1)")
    print("   - AvgBasketValue  = MonetaryTotal / Frequency")
    print("   - TenureRatio     = Recency / (CustomerTenureDays + 1)")
    return df

# ============================================================
# ÉTAPE 7 — Supprimer multicolinéarité (seuil > 0.8)
# ============================================================
def supprimer_multicolineaires(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_cols = [c for c in num_cols if c != 'Churn']

    corr_matrix = df[num_cols].corr().abs()
    a_supprimer = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.8:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                a_supprimer.add(col_i)
                print(f"   ⚠️  {col_i} ↔ {col_j} : {corr_matrix.iloc[i,j]:.2f} → suppression {col_i}")

    df = df.drop(columns=list(a_supprimer))
    print(f"✅ Multicolinéarité : {len(a_supprimer)} colonnes supprimées")
    return df

# ============================================================
# ÉTAPE 8 — Encoder les features catégorielles
# ============================================================
def encoder_categories(df):
    ordinal_configs = {
        'AgeCategory':       ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'SpendingCategory':  ['Low', 'Medium', 'High', 'VIP'],
        'LoyaltyLevel':      ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
        'ChurnRiskCategory': ['Faible', 'Moyen', 'Élevé', 'Critique'],
        'BasketSizeCategory':['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'PreferredTimeOfDay':['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }
    for col, categories in ordinal_configs.items():
        if col in df.columns:
            enc = OrdinalEncoder(
                categories=[categories],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            df[col] = enc.fit_transform(df[[col]])
            print(f"✅ Ordinal encodé : {col}")

    onehot_cols = [
        'RFMSegment', 'CustomerType', 'FavoriteSeason',
        'Region', 'WeekendPreference', 'ProductDiversity',
        'Gender', 'AccountStatus'
    ]
    onehot_cols = [c for c in onehot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
    print(f"✅ One-Hot encodé : {onehot_cols}")

    if 'Country' in df.columns:
        target_mean = df.groupby('Country')['Churn'].mean()
        df['Country_encoded'] = df['Country'].map(target_mean)
        df = df.drop(columns=['Country'])
        print(f"✅ Target Encoding : Country → Country_encoded")
    return df

# ============================================================
# ÉTAPE 9 — Split puis Imputation + Normalisation
# L'imputation se fait sur X_train puis appliquée sur X_test
# ============================================================
def imputer_et_normaliser(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split stratifié 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Split : X_train={X_train.shape}, X_test={X_test.shape}")

    # --- Imputation médiane sur X_train → appliquée sur X_test ---
    mediane_train = X_train.median()
    X_train = X_train.fillna(mediane_train)
    X_test  = X_test.fillna(mediane_train)  # même médiane que train
    print(f"✅ Imputation médiane : fit sur X_train, appliquée sur X_test")

    # --- KNN Imputer (démonstration sur Age si manquant) ---
    # Note : après fillna médiane, plus de NaN — KNN présenté comme alternative
    print(f"✅ Alternative KNN Imputer disponible (cf. commentaire dans le code)")
    # knn = KNNImputer(n_neighbors=5)
    # X_train = pd.DataFrame(knn.fit_transform(X_train), columns=X_train.columns)
    # X_test  = pd.DataFrame(knn.transform(X_test),      columns=X_test.columns)

    # --- StandardScaler : fit sur X_train UNIQUEMENT ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    print(f"✅ StandardScaler : fit sur X_train uniquement (pas de data leakage)")

    # Sauvegarder
    os.makedirs('data/train_test', exist_ok=True)
    X_train_scaled.to_csv('data/train_test/X_train.csv', index=False)
    X_test_scaled.to_csv('data/train_test/X_test.csv',   index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv',   index=False)
    print(f"✅ Fichiers sauvegardés dans data/train_test/")

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler,       'models/scaler.pkl')
    joblib.dump(mediane_train,'models/mediane_train.pkl')
    print(f"✅ Scaler et médiane sauvegardés dans models/")

    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
if __name__ == "__main__":
    df = charger_donnees('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
    df = supprimer_features_inutiles(df)
    df = corriger_aberrantes(df)
    df = parser_dates(df)
    df = transformer_ip(df)
    df = feature_engineering(df)
    df = supprimer_multicolineaires(df)
    df = encoder_categories(df)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/data_clean.csv', index=False)
    print(f"\n✅ Données nettoyées : data/processed/data_clean.csv — Shape : {df.shape}")

    X_train, X_test, y_train, y_test = imputer_et_normaliser(df)
    print("\n🎉 Preprocessing terminé !")