import pandas as pd
import numpy as np
import re
import ipaddress
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

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
    df = df.drop(columns=cols_a_supprimer)
    print(f"✅ Features supprimées : {cols_a_supprimer}")
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
# ÉTAPE 4 — Imputer les valeurs manquantes
# ============================================================
def imputer_manquantes(df):
    mediane_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(mediane_age)
    print(f"✅ Age imputé avec médiane = {mediane_age:.1f}")

    mediane_days = df['AvgDaysBetweenPurchases'].median()
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(mediane_days)
    print(f"✅ AvgDaysBetweenPurchases imputé avec médiane = {mediane_days:.1f}")

    for col in ['SupportTicketsCount', 'SatisfactionScore']:
        mediane = df[col].median()
        df[col] = df[col].fillna(mediane)
        print(f"✅ {col} imputé avec médiane = {mediane:.1f}")
    return df

# ============================================================
# ÉTAPE 5 — Parser RegistrationDate
# ============================================================
def parser_dates(df):
    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'],
        dayfirst=True,
        errors='coerce'
    )
    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday
    df = df.drop(columns=['RegistrationDate'])
    print(f"✅ RegistrationDate parsée → RegYear, RegMonth, RegDay, RegWeekday")
    return df

# ============================================================
# ÉTAPE 6 — Transformer LastLoginIP
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
# ÉTAPE 7 — Feature Engineering (section 6.1 cahier des charges)
# ============================================================
def feature_engineering(df):
    # Ratio dépenses / recency
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # Panier moyen
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']

    # Ancienneté vs activité récente
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    print("✅ Feature Engineering :")
    print("   - MonetaryPerDay  = MonetaryTotal / (Recency + 1)")
    print("   - AvgBasketValue  = MonetaryTotal / Frequency")
    print("   - TenureRatio     = Recency / (CustomerTenureDays + 1)")
    return df

# ============================================================
# Supprimer multicolinéarité (seuil > 0.8) — section 6.1
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
# ÉTAPE 9 — Normaliser + Split train/test + Sauvegarder
# ============================================================
def normaliser_et_splitter(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Split : X_train={X_train.shape}, X_test={X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    print(f"✅ StandardScaler : fit sur X_train uniquement (pas de data leakage)")

    os.makedirs('data/train_test', exist_ok=True)
    X_train_scaled.to_csv('data/train_test/X_train.csv', index=False)
    X_test_scaled.to_csv('data/train_test/X_test.csv',   index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv',   index=False)
    print(f"✅ Fichiers sauvegardés dans data/train_test/")

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"✅ Scaler sauvegardé dans models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================================
# PIPELINE PRINCIPAL — exécuter tout dans l'ordre
# ============================================================
if __name__ == "__main__":
    df = charger_donnees('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
    df = supprimer_features_inutiles(df)
    df = corriger_aberrantes(df)
    df = imputer_manquantes(df)
    df = parser_dates(df)
    df = transformer_ip(df)
    df = feature_engineering(df)
    df = supprimer_multicolineaires(df)
    df = encoder_categories(df)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/data_clean.csv', index=False)
    print(f"\n✅ Données nettoyées sauvegardées : data/processed/data_clean.csv")
    print(f"   Shape finale : {df.shape}")

    X_train, X_test, y_train, y_test = normaliser_et_splitter(df)
    print("\n🎉 Preprocessing terminé !")