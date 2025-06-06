import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier, plot_importance, to_graphviz
import re

# 1. Wczytanie i przygotowanie danych
df = pd.read_csv('data/data.csv')
df = df.drop(columns=['Unnamed: 32', 'id'])

label_encoder = LabelEncoder()
df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])
X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
y = df['diagnosis_encoded']

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Trenowanie modelu XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 4. Predykcje i obliczenie metryk
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

accuracy = round(accuracy_score(y_test, y_pred), 4)
f1 = round(f1_score(y_test, y_pred), 4)
auc = round(roc_auc_score(y_test, y_pred_proba), 4)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"AUC: {auc}")

float_pattern = re.compile(r'(-?\d+\.\d+)')  # znajdzie wszystkie wystąpienia liczb zmiennoprzecinkowych
dumped_trees = xgb_model.get_booster().get_dump()  
n_trees = len(dumped_trees)
for i in range(n_trees):
    dot = to_graphviz(xgb_model, num_trees=i, rankdir='UD')
    src = dot.source
    
    # Zamieniamy wszystkie liczby zmiennoprzecinkowe na wartość zaokrągloną do 4 miejsc
    src_rounded = float_pattern.sub(lambda m: f"{float(m.group(1)):.4f}", src)
    
    # Zapisujemy zmodyfikowany tekst DOT do pliku
    with open(f"results/trees/dot/tree_{i}.dot", 'w') as f:
        f.write(src_rounded)

# 5. Wizualizacja krzywej ROC
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, label=f"AUC = {auc}")
# plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for XGBoost Classifier')
# plt.legend(loc='lower right')
# plt.show()

# 6. Wizualizacja istotności cech
# plt.figure(figsize=(8, 10))
# plot_importance(xgb_model, max_num_features=10, height=0.6)
# plt.title('Top 10 Feature Importances')
# plt.show()