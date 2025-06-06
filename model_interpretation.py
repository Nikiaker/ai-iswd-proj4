import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import matplotlib.pyplot as plt

# 1. Wczytanie i przygotowanie danych
df = pd.read_csv('data/data.csv')
df = df.drop(columns=['Unnamed: 32', 'id'])

label_encoder = LabelEncoder()
df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])
X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
y = df['diagnosis_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Trenowanie modelu XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 3. PERMUTATION FEATURE IMPORTANCE na zbiorze testowym
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

perm_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)

print("Permutation Feature Importance (średnia ± odchylenie):")
print(perm_importance_df.to_string(index=False))

# 4. PARTIAL DEPENDENCE PLOTS (PDP) dla wybranych cech
#    Wybieramy kilka kluczowych cech, które w poprzednich analizach okazały się najważniejsze:
features_to_plot = [
    'perimeter_worst',
    'texture_worst',
    'concave points_worst',
    'area_se',
    'concave points_mean'
]

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
ax = ax.flatten()

for i, feat in enumerate(features_to_plot):
    # kind='average' oznacza typowy PDP (średnia z ICE)
    PartialDependenceDisplay.from_estimator(model, X_train, [feat], ax=ax[i], kind='average')
    ax[i].set_title(f'PDP dla "{feat}"')
    ax[i].set_ylabel('Predykowane prawdopodobieństwo M')

# Usuń pusty subplot, jeśli jest
if len(features_to_plot) < len(ax):
    fig.delaxes(ax[-1])

plt.tight_layout()
plt.show()
