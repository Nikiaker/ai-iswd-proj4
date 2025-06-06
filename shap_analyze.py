import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap

# 1.1. Wczytanie i przygotowanie całego zbioru danych, tak jak dotychczas
df = pd.read_csv('data/data.csv')  # dopasuj ścieżkę do Twojego pliku
df = df.drop(columns=['Unnamed: 32', 'id'])

label_encoder = LabelEncoder()
df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])
X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
y = df['diagnosis_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 1.2. Przygotowanie trzech wariantów jako DataFrame (bez kolumn "id" i "diagnosis_encoded")
variants = pd.read_csv('data/selected_variants.csv')
variants = variants.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])

# 1.3. Utworzenie obiektu SHAP Explainer (TreeExplainer dla XGBoost)
explainer = shap.TreeExplainer(model)

# 1.4. Obliczenie wartości SHAP dla trzech wariantów
shap_values = explainer.shap_values(variants)

# 1.5. Zamiana predykcji na etykiety (0=B, 1=M)
preds = model.predict(variants)

# 1.6. Przygotowanie przejrzystego DataFrame z najważniejszymi 5 cechami dla każdego wariantu
feature_names = variants.columns.tolist()

topk = 5  # liczba cech do wypisania w kolejności malejącego |shap|
results = []

for i, sample in variants.iterrows():
    # Dla danego wiersza (i) pobierz shap_values[i] (wektor długości liczba cech)
    sv = shap_values[i]
    # Stwórz DataFrame z dwoma kolumnami: cecha, shap_value
    df_sv = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv
    })
    # Posortuj po |shap_value| malejąco i weź topk
    df_sv["abs_shap"] = df_sv["shap_value"].abs()
    df_top = df_sv.sort_values("abs_shap", ascending=False).head(topk).drop(columns="abs_shap")
    
    results.append({
        "id": [865137, 884948, 901303][i],
        "pred_before": "M" if preds[i] == 1 else "B",
        "top_features": df_top
    })

# 1.7. Wyświetlenie wyników
for r in results:
    print(f"Wariant ID = {r['id']}, model przewiduje klasę = {r['pred_before']}")
    print("Top 5 cech wg |SHAP| (wartość SHAP, dodatnia ⇒ pcha w kierunku 'M', ujemna ⇒ pcha w kierunku 'B'):")
    print(r["top_features"].reset_index(drop=True))
    print("\n" + "-"*80 + "\n")
