import pandas as pd

df = pd.read_csv('data/processed/data_clean.csv')

print("=== COLONNES CORRÉLÉES AVEC CHURN (> 0.5) ===")
for col in df.columns:
    if col != 'Churn':
        try:
            corr = abs(df[col].corr(df['Churn']))
            if corr > 0.5:
                print(f"{col} : {corr:.3f}")
        except:
            pass

print("\n=== TOUTES LES CORRÉLATIONS TRIÉES ===")
corrs = {}
for col in df.columns:
    if col != 'Churn':
        try:
            corr = abs(df[col].corr(df['Churn']))
            corrs[col] = round(corr, 3)
        except:
            pass

for col, val in sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"{col} : {val}")