#Entrenamiento multi-modelo
#Entrena múltiples modelos sobre el dataset Bank Marketing.
#lirerias e importaciones
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm  as lgb
import catboost as cb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import get_features, build_preprocessing_pipeline

#Cargar dataset y separa X, y eliminando 'duration' (leakage).
def load_data(path: str = '../data/raw/bank_marketing.csv'):
    df = pd.read_csv(path)
    y = df['Class'].map({'yes': 1, 'no': 0})
    X = df.drop(columns=['Class', 'duration'])
    return X, y 

#Entrenar y evalúar múltiples modelos
def evaluate_models(model, X_train, y_train, X_test, y_test):
    model.fit()(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)[:, 1]
    
    metrics = {
        'ROC-AUC': roc_auc_score(y_test, y_pred),
        'F1': classification_report(y_test, y_pred, output_dict= True)['weighted avg']['f1-score'],
        'Accuracy': classification_report(y_test, y_pred, output_dict=True)['accuracy'],
        'Recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
    }
    return metrics

#Cargar datos
def run_experiments():
    X, y = load_data()
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    #Features
    num_features, cat_features = get_features(pd.concat([X, y], axis=1), target='Class', drop=['duration'])

    #Preprocesador con SMOTE
    preprocessor = build_preprocessing_pipeline(num_features, cat_features, use_smote=True)
    X_train_prep, y_train_prep = preprocessor.fit_resample(X_train, y_train)
    X_test_prep = preprocessor.transform(X_test)
    
    #Modelos
    models = {
        'LogReg': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric = 'logloss', random_state=42),
        'lightGBM': lgb.LGBMClassifier(random_state=42),
        'catBoost': cb.CatBoostClassifier(verbose=0, random_state=42)
    }
    
    #Resultados
    results = {}
    for name, model in models.items():
        print(f'\nEntrenando {name}...')
        metrics = evaluate_models(model, X_train_prep, y_train_prep, X_test_prep, y_test)
        results[name] = metrics
        print(f'{name} → ROC-AUC: {metrics['ROC-AUC']:.3f} | F1: {metrics['F1']:.3f}')
    return results

if __name__ == '__main__':
    results = run_experiments()
    print('\n=== Resultados finales ===')
    for model, metrics in results.items():
        print(model, metrics)