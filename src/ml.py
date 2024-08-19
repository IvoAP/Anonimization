import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from anonimization import anonimization_clustering


def cross_validate_k_fold(X, y, anon_training, anon_test, model, model_name, n_clusters):
    kf = StratifiedKFold(n_splits=3)
    scaler = StandardScaler()

    accuracy, precision, recall, f1 = [], [], [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if anon_training:
            X_train, y_train = anonimization_clustering(X_train, y_train, n_clusters)

        if anon_test:
            X_test, y_test = anonimization_clustering(X_test, y_test, n_clusters)

        # Normalizar os dados
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        f1.append(f1_score(y_test, y_pred, average='weighted'))

    results = {
        'accuracy': np.array(accuracy), 
        'precision': np.array(precision),
        'recall': np.array(recall), 
        'f1_score': np.array(f1)
    }

    print(model_name, anon_training, anon_test)
    for k in results.keys():
        print(f"{k} ---> mean: {results[k].mean():.4f}, std: {results[k].std():.4f}")

    return [anon_training, anon_test, results['accuracy'].mean(), results['precision'].mean(), results['recall'].mean(), results['f1_score'].mean()]

def find_best_results(results, feature_selections, feature_method, num_features):
    best_results = []

    scenarios = ['True_True', 'True_False', 'False_True', 'False_False']
    for model_name in results['model'].unique():
        for scenario in scenarios:
            scenario_split = scenario.split('_')
            anonymized_train = scenario_split[0] == 'True'
            anonymized_test = scenario_split[1] == 'True'
            model_results = results[
                (results['model'] == model_name) &
                (results['anonymized train'] == anonymized_train) &
                (results['anonymized test'] == anonymized_test)
            ]
            if not model_results.empty:
                best_result = model_results.loc[model_results['accuracy'].idxmax()]
                best_results.append(best_result.to_list())

    return best_results

