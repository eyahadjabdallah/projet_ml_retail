import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/processed/data_clean.csv')

X = df.drop(columns=['Churn'])
y = df['Churn']

# Supprimer tout ce qu'on a identifié jusqu'ici
cols_leakage = [c for c in X.columns
                if 'ChurnRisk' in c
                or 'CustomerType_Perdu' in c
                or 'RFMSegment_Dormants' in c
                or 'FavoriteSeason_Automne' in c]

print(f"Colonnes supprimées : {cols_leakage}")
X = X.drop(columns=cols_leakage, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.3f}")

importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(15)

print("\n=== TOP 15 FEATURES ===")
for col, val in importances.items():
    print(f"  {col} : {val:.4f}")