import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/processed/data_clean.csv')

# Test 1 — seulement les 5 features RFM de base
features_simples = ['Recency', 'Frequency', 'MonetaryTotal', 'Age', 'CustomerTenureDays']
X_simple = df[features_simples]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Accuracy avec 5 features seulement : {acc:.3f}")

# Test 2 — toutes les features sauf leakage connu
X_full = df.drop(columns=['Churn', 'ChurnRiskCategory', 'CustomerType_Perdu'],
                 errors='ignore')
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y)

lr2 = LogisticRegression(random_state=42, max_iter=1000)
lr2.fit(X_train2, y_train2)
acc2 = accuracy_score(y_test2, lr2.predict(X_test2))
print(f"Accuracy avec toutes les features  : {acc2:.3f}")

# Afficher la différence
print(f"\nSi Test1 << Test2 → il y a encore une colonne qui triche")
print(f"Si Test1 ≈ Test2  → le problème vient des données brutes elles-mêmes")