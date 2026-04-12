import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ============================================================
# 1. Chargement
# ============================================================
def charger_donnees(chemin):
    df = pd.read_csv(chemin)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

# ============================================================
# 2. Suppression des colonnes inutiles
# ============================================================
def supprimer_colonnes_inutiles(df):
    cols_inutiles = ['NewsletterSubscribed', 'CustomerID']
    df = df.drop(columns=cols_inutiles, errors='ignore')
    print(f"✅ Supprimées (inutiles) : {cols_inutiles}")
    return df

# ============================================================
# 3. Correction valeurs aberrantes
# ============================================================
def corriger_aberrantes(df):
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace({999: np.nan, -1: np.nan})
    df['SatisfactionScore'] = df['SatisfactionScore'].replace({99: np.nan, -1: np.nan})
    print("✅ SupportTicketsCount et SatisfactionScore : valeurs aberrantes → NaN")
    return df

# ============================================================
# 4. Parsing RegistrationDate
# ============================================================
def parser_dates(df):
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear'] = df['RegistrationDate'].dt.year
    df['RegMonth'] = df['RegistrationDate'].dt.month
    df['RegDay'] = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday
    df = df.drop(columns=['RegistrationDate'])
    print("✅ RegistrationDate parsée → RegYear, RegMonth, RegDay, RegWeekday")
    return df

# ============================================================
# 5. IP → IsPrivateIP
# ============================================================
def transformer_ip(df):
    def is_private(ip_str):
        try:
            return int(ipaddress.ip_address(str(ip_str)).is_private)
        except:
            return 0
    df['IsPrivateIP'] = df['LastLoginIP'].apply(is_private)
    df = df.drop(columns=['LastLoginIP'])
    print("✅ LastLoginIP → IsPrivateIP")
    return df

# ============================================================
# 6. Feature engineering (utilise CustomerTenureDays et Recency)
# ============================================================
def feature_engineering(df):
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    print("✅ Feature engineering : MonetaryPerDay, AvgBasketValue, TenureRatio")
    return df

# ============================================================
# 7. Suppression des colonnes de leakage (après feature engineering)
# ============================================================
def supprimer_colonnes_leakage(df):
    cols_leakage = [
        'ChurnRiskCategory', 'CustomerType', 'LoyaltyLevel',
        'SpendingCategory', 'RFMSegment', 'AccountStatus',
        'ReturnRatio', 'NegQtyCount', 'ZeroPriceCount',
        'CancelledTransactions',
        'CustomerTenureDays', 'FirstPurchase', 'Age',
        'SupportTicketsCount', 'SatisfactionScore',
        'Recency',          # <--- AJOUT (corrélation 0.86 avec Churn)
        'TenureRatio'       # <--- AJOUT (dérivé de Recency)
    ]
    df = df.drop(columns=cols_leakage, errors='ignore')
    print(f"✅ Supprimées (leakage) : {cols_leakage}")
    return df

# ============================================================
# 8. Suppression multicolinéarité (seuil > 0.8)
# ============================================================
def supprimer_multicolineaires(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_cols = [c for c in num_cols if c != 'Churn']
    if len(num_cols) == 0:
        print("⚠️ Aucune colonne numérique pour la corrélation")
        return df
    corr_matrix = df[num_cols].corr().abs()
    a_supprimer = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.8:
                col_i = corr_matrix.columns[i]
                a_supprimer.add(col_i)
                print(f"   ⚠️  {col_i} ↔ {corr_matrix.columns[j]} : {corr_matrix.iloc[i,j]:.2f} → suppression {col_i}")
    df = df.drop(columns=list(a_supprimer))
    print(f"✅ Multicolinéarité : {len(a_supprimer)} colonnes supprimées")
    return df

# ============================================================
# 9. Encodage catégoriel (sans Country)
# ============================================================
def encoder_categories_sans_country(df):
    # Ordinal
    ordinal_configs = {
        'AgeCategory': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
        'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
        'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
    }
    for col, cats in ordinal_configs.items():
        if col in df.columns:
            enc = OrdinalEncoder(categories=[cats], handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = enc.fit_transform(df[[col]])
            print(f"✅ Ordinal encodé : {col}")

    # One-Hot (sauf Country)
    onehot_cols = ['FavoriteSeason', 'Region', 'WeekendPreference', 'ProductDiversity', 'Gender']
    onehot_cols = [c for c in onehot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
    print(f"✅ One-Hot encodé : {onehot_cols}")
    return df

# ============================================================
# 10. Split + One-Hot de Country + Imputation + Scaling
# ============================================================
def preparer_train_test(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Split : X_train={X_train.shape}, X_test={X_test.shape}")

    # One-Hot encoding de Country (après split, sans fuite)
    if 'Country' in X_train.columns:
        combined = pd.concat([X_train[['Country']], X_test[['Country']]], axis=0)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(combined[['Country']])
        X_train_ohe = pd.DataFrame(ohe.transform(X_train[['Country']]), 
                                   columns=[f'Country_{c}' for c in ohe.categories_[0]])
        X_test_ohe = pd.DataFrame(ohe.transform(X_test[['Country']]),
                                  columns=[f'Country_{c}' for c in ohe.categories_[0]])
        X_train = X_train.drop(columns=['Country']).reset_index(drop=True)
        X_test = X_test.drop(columns=['Country']).reset_index(drop=True)
        X_train = pd.concat([X_train, X_train_ohe], axis=1)
        X_test = pd.concat([X_test, X_test_ohe], axis=1)
        print(f"✅ One-Hot encoding de Country : {len(ohe.categories_[0])} pays")
    else:
        print("⚠️  Colonne 'Country' absente")

    # Imputation médiane
    mediane_train = X_train.median()
    X_train = X_train.fillna(mediane_train)
    X_test = X_test.fillna(mediane_train)
    print("✅ Imputation médiane (train → test)")

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    print("✅ StandardScaler (fit sur train)")

    # Sauvegardes
    os.makedirs('data/train_test', exist_ok=True)
    X_train_scaled.to_csv('data/train_test/X_train.csv', index=False)
    X_test_scaled.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(mediane_train, 'models/mediane_train.pkl')
    print("✅ Fichiers sauvegardés dans data/train_test/ et models/")

    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================================
# Pipeline principal
# ============================================================
if __name__ == "__main__":
    df = charger_donnees('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
    df = supprimer_colonnes_inutiles(df)
    df = corriger_aberrantes(df)
    df = parser_dates(df)
    df = transformer_ip(df)
    df = feature_engineering(df)
    df = supprimer_colonnes_leakage(df)   # suppression après feature engineering
    df = supprimer_multicolineaires(df)
    df = encoder_categories_sans_country(df)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/data_clean.csv', index=False)
    print(f"\n✅ data_clean.csv sauvegardé — Shape : {df.shape}")

    X_train, X_test, y_train, y_test = preparer_train_test(df)
    print("\n🎉 Preprocessing terminé !")