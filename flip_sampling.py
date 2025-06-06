import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 1. Wczytanie i trenowanie modelu
df = pd.read_csv('data/data.csv')      # dopasuj ścieżkę do pliku data.csv
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

# 2. „Trzy warianty” jako pandas.DataFrame
variants = pd.read_csv('data/selected_variants.csv')

# Lista cech, które trafiają do modelu (bez "id" i "diagnosis")
features = [col for col in variants.columns if col not in ["id", "diagnosis", "Unnamed: 32"]]

# 3. Funkcja do próbkowania i wykrywania momentu, w którym klasa się zmienia
def find_flip_threshold(model, base_sample, feature, direction, num_steps=500):
    """
    model        : wytrenowany XGBClassifier
    base_sample  : pandas.Series z oryginalnymi cechami próbki
    feature      : nazwa pojedynczej cechy, którą będziemy zmieniać
    direction    : +1 (podkręcamy w górę) lub -1 (obniżamy)
    num_steps    : liczba punktów dyskretnych do sprawdzenia w zadanym przedziale
    """
    original_value = base_sample[feature]
    base_pred = model.predict(pd.DataFrame([base_sample[features]]))[0]
    base_proba = model.predict_proba(pd.DataFrame([base_sample[features]]))[0][1]  # Prawdopodobieństwo klasy 1 (M)
    # Ustalamy zakres: od oryginalnej wartości do ±50% lub –80%, w zależności od kierunku
    if direction > 0:
        end_value = original_value * (100 + 0.5)   # do +50% (można zmodyfikować)
    else:
        end_value = original_value * (100 - 0.8)   # do –80% (można zmodyfikować)
    # Generujemy siatkę num_steps półotwartych przedziałów
    candidate_values = np.linspace(original_value, end_value, num_steps)
    # Próbkujemy kolejno i sprawdzamy, kiedy model przewidzi inną klasę
    for val in candidate_values[1:]:
        tmp = base_sample.copy()
        tmp[feature] = val
        pred = model.predict(pd.DataFrame([tmp[features]]))[0]
        pred_proba = model.predict_proba(pd.DataFrame([tmp[features]]))[0][1]  # Prawdopodobieństwo klasy 1 (M)
        if pred != base_pred:
            return val, base_pred, pred
    return end_value, base_proba, pred_proba

# 4. Dla każdego wariantu wykonujemy „próbkowanie” tylko tej jednej cechy:
results = []

# Wariant 1: "texture_worst" zwiększamy (+1)
v1 = variants.iloc[0]
flip_v1, before_v1, after_v1 = find_flip_threshold(model, v1, "texture_worst", direction=1)

# Wariant 2: "concave points_worst" zmniejszamy (-1)
v2 = variants.iloc[1]
flip_v2, before_v2, after_v2 = find_flip_threshold(model, v2, "concave points_worst", direction=-1)

# Wariant 3: "concave points_mean" zwiększamy (+1)
v3 = variants.iloc[2]
flip_v3, before_v3, after_v3 = find_flip_threshold(model, v3, "concave points_mean", direction=1)

# Zbieramy wyniki
results.append({
    "id": v1["id"],
    "feature": "texture_worst",
    "original_value": v1["texture_worst"],
    "flip_value": flip_v1,
    "pred_before": before_v1,
    "pred_after": after_v1
})
results.append({
    "id": v2["id"],
    "feature": "concave points_worst",
    "original_value": v2["concave points_worst"],
    "flip_value": flip_v2,
    "pred_before": before_v2,
    "pred_after": after_v2
})
results.append({
    "id": v3["id"],
    "feature": "concave points_mean",
    "original_value": v3["concave points_mean"],
    "flip_value": flip_v3,
    "pred_before": before_v3,
    "pred_after": after_v3
})

# Konwersja na DataFrame do czytelnego wyświetlenia
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
