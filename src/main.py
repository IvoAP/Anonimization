import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (SelectFromModel, SelectKBest,
                                       VarianceThreshold, chi2, f_classif,
                                       mutual_info_classif)
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from file_utils import get_file
from ml import cross_validate_k_fold


# Function that select the technique of feature selection
def feature_selection(X, y, method, k=None):
    if method == 'chi2':
        selector = SelectKBest(chi2, k= k)
    elif method == 'lasso':
        alpha = 0.01 # This need to adjust to be the best lasso
        lasso = Lasso(alpha=alpha)
        lasso.fit(X,y)
        selector = SelectFromModel(lasso, prefit=True)
    elif method == 'low_variance':
        threshold = 0.1  # Ajuste conforme necessário
        selector = VarianceThreshold(threshold)
    elif method == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True)
    elif method == 'anova':
        selector = SelectKBest(f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    else:
        raise ValueError(f"Método de seleção de características não suportado: {method}")

    X_new = selector.fit_transform(X,y)
    selected_features_idx =  selector.get_support(indices=True)
    return X_new, selected_features_idx

# Function to  get the results
def get_result(model, X, y, model_name, n_clusters, feature_method, k):
    bol = [True, False]
    results_columns = ['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score']
    results = pd.DataFrame(columns=results_columns)
    selected_features_all = []

    for i in range(0, 2):
        for j in range(0, 2):
            # Feature Selection
            X_new, selected_features_idx = feature_selection(X, y, feature_method, k)
            selected_features_all.append({
                'anonymized_train': bol[i],
                'anonymized_test': bol[j],
                'model': model_name,
                'feature_method': feature_method,
                'num_features': k,
                'selected_features_idx': selected_features_idx.tolist()
            })
            cross_val_results = cross_validate_k_fold(X_new, y, bol[i], bol[j], model, model_name, n_clusters)
            new_df = pd.DataFrame([[
                model_name, 
                bol[i], 
                bol[j], 
                cross_val_results[2], 
                cross_val_results[3], 
                cross_val_results[4], 
                cross_val_results[5]
            ]], columns=results_columns)
            results = pd.concat([results, new_df], ignore_index=True)

    return results, selected_features_all

# Function where the experiment are maded
def experiment(X, y, feature_method, k):
    all_results = pd.DataFrame(columns=['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score', 'selected_features', 'feature_method', 'num_features'])
    models = [
        (KNeighborsClassifier(n_neighbors=5), 'KNN'),
        (DecisionTreeClassifier(), 'Decision Tree'),
        (RandomForestClassifier(n_estimators=100), 'Random Forest'),
        (GaussianNB(var_smoothing=1e-02), 'GaussianNB'),
        (MLPClassifier(
            hidden_layer_sizes=(100, 50),  
            activation='relu',             
            solver='adam',                 
            alpha=0.0001,                 
            learning_rate='adaptive',     
            learning_rate_init=0.001,      
            max_iter=500,                  
            early_stopping=True,           
            validation_fraction=0.1
        ), 'Mutilayer Perceptron')
        
    ]

    for model, model_name in models:
        results, selected_features = get_result(model, X, y, model_name, 3, feature_method, k)
        # Find the best resutlt for which scenario
        best_results = find_best_results(results, selected_features, feature_method, k)
        all_results = pd.concat([all_results, best_results], ignore_index=True)

    return all_results

def find_best_results(results, selected_features, feature_method, k):
    scenarios = [(True, True), (True, False), (False, True), (False, False)]
    best_results = []

    for model_name in results['model'].unique():
        for scenario in scenarios:
            anonymized_train, anonymized_test = scenario
            model_results = results[
                (results['model'] == model_name) &
                (results['anonymized_train'] == anonymized_train) &
                (results['anonymized_test'] == anonymized_test)
            ]
            if not model_results.empty:
                best_result = model_results.loc[model_results['accuracy'].idxmax()]
                selected_feature_info = [s for s in selected_features if 
                                          s['anonymized_train'] == anonymized_train and
                                          s['anonymized_test'] == anonymized_test and
                                          s['model'] == model_name and
                                          s['feature_method'] == feature_method and
                                          s['num_features'] == k]
                if selected_feature_info:
                    best_result['selected_features'] = selected_feature_info[0]['selected_features_idx']
                best_result['feature_method'] = feature_method
                best_result['num_features'] = k
                best_results.append(best_result)

    return pd.DataFrame(best_results)

def Chi2(X, y):
    all_best_results = []

    for i in range(2,12,1):
        
        best_results = experiment(X, y, 'chi2', i)
        all_best_results.append(best_results)
        print(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    final_best_results_df.to_csv('best_results_chi2.csv', index=False)
    print("Melhores resultados salvos em 'best_results_chi2.csv'")
    print(final_best_results_df.head())

def Lasso(X, y):
    results, selected_features = experiment(X, y, 'lasso', None)
    
    best_results = find_best_results(results, selected_features, 'lasso', None)
    best_results.to_csv('best_results_lasso.csv', index=False)

def LowVariance(X, y):
    results, selected_features = experiment(X, y, 'low_variance', None)
    
    best_results = find_best_results(results, selected_features, 'low_variance', None)
    best_results.to_csv('best_results_low_variance.csv', index=False)

def ExtraTree(X, y):
    all_best_results = []

    for i in range(3, 80, 5):
        best_results = experiment(X, y, 'extra_trees', i)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    final_best_results_df.to_csv('best_results_extra_trees.csv', index=False)

def ANOVA(X, y):
    all_best_results = []

    for i in range(2, 10):
        best_results = experiment(X, y, 'anova', i)
        all_best_results.append(best_results)

    for i in range(10, 80, 5):
        best_results = experiment(X, y, 'anova', i)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    final_best_results_df.to_csv('best_results_anova.csv', index=False)

def MutualInformation(X, y):
    all_best_results = []

    for i in range(2, 10):
        best_results = experiment(X, y, 'mutual_info', i)
        all_best_results.append(best_results)

    for i in range(10, 80, 5):
        best_results = experiment(X, y, 'mutual_info', i)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    final_best_results_df.to_csv('best_results_mutual_info.csv', index=False)


def main():
    # Set the renadom seed 
    np.random.seed(7) 

    # read the dataset
    dataset = get_file()
    print(len(dataset.columns))

    # Extract feature and label
    y = np.array(dataset['Label'])
    del dataset['Label']
    X = np.array(dataset)

  # Escolher o método de seleção de características a ser usado
    Chi2(X, y)
    # Lasso(X, y)
    # LowVariance(X, y)
    # ExtraTree(X, y)
    # ANOVA(X, y)
    # MutualInformation(X, y)


if __name__ == "__main__":
    main()
