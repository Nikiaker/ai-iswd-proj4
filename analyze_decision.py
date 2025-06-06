import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, DMatrix
from tabulate import tabulate

def get_variants(variants: pd.DataFrame, xgb_model: XGBClassifier):
    features_variants = variants.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
    dmat_var = DMatrix(features_variants)
    contribs = xgb_model.get_booster().predict(dmat_var, pred_contribs=True)

    probs = xgb_model.predict_proba(features_variants)[:,1]
    preds = xgb_model.predict(features_variants)

    # Collect top contributions
    feat_names = features_variants.columns.tolist()
    results = []
    for i in range(variants.shape[0]):
        contrib_dict = dict(zip(feat_names + ['bias'], contribs[i]))
        sorted_contrib = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        results.append({
            'id': variants.loc[i,'id'],
            'true': variants.loc[i,'diagnosis'],
            'pred': 'M' if preds[i]==1 else 'B',
            'proba': round(probs[i],4),
            'top_features': [(feat, round(val,4)) for feat, val in sorted_contrib]
        })
    
    return results

def print_results(results: list):
    for res in results:
        print(f"id: {res['id']}")
        print(f"True Diagnosis: {res['true']}")
        print(f"Predicted Diagnosis: {res['pred']}")
        print(f"Probability: {res['proba']}")
        print("Top Features:")
        for feat, val in res['top_features']:
            print(f"  {feat}: {val}")

def main():
    # Load full dataset for training
    df = pd.read_csv('data/data.csv')
    df = df.drop(columns=['Unnamed: 32', 'id'])

    label_encoder = LabelEncoder()
    df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])
    X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
    y = df['diagnosis_encoded']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Define variants
    variants = pd.read_csv('data/selected_variants.csv')
    results = get_variants(variants, xgb_model)
    #print_results(results)

    changed_variants = pd.read_csv('data/selected_variants_vals_changed.csv')
    changed_results = get_variants(changed_variants, xgb_model)
    print_results(changed_results)


    # # Print results in a nice table
    # table = []
    # for result in results:
    #     table.append([
    #         result['id'],
    #         result['true'],
    #         result['pred'],
    #         result['proba'],
    #         ", ".join([f"{feat} ({val})" for feat, val in result['top_features']])
    #     ])

    # headers = ["ID", "True Diagnosis", "Predicted Diagnosis", "Probability", "Top Features"]
    # print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()