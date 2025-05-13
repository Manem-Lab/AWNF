#!/usr/bin/env python
# coding: utf-8


from adaptive_weighted_snf import *
import snf
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import snf
from ml_funcs import  *
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.utils.validation import (check_array, check_symmetric,
                                      check_consistent_length)
import json
from sklearn.metrics import roc_auc_score

def drop_highly_correlated(X, target,column_name, threshold=0.9):
    """
    Drops highly correlated features but retains the ones best correlated with the target.
    
    :param df: Pandas DataFrame
    :param target_col: Name of the target column
    :param threshold: args.snf_metric threshold for dropping features
    :return: Reduced DataFrame
    """
    df=pd.concat([X,target], axis=1)
    corr_matrix = df.corr(method='pearson')  # Compute args.snf_metric matrix
    
    # Identify features correlated with the target
    target_corr = corr_matrix[column_name].abs().sort_values(ascending=False)

    # Mask to find highly correlated feature pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns that are highly correlated
    to_drop = set()
    for column in upper_tri.columns:
        high_corr_features = upper_tri[column][upper_tri[column] > threshold].index.tolist()
        for feature in high_corr_features:
            # Drop the feature that is less correlated with the target
            if target_corr[feature] < target_corr[column]:
                to_drop.add(feature)
            else:
                to_drop.add(column)

    return to_drop


def perform_grid_search(xrna_train, y_train, param_range=np.arange(0.1, 1.0, 0.1), cv_folds=5):
    """
    Perform GridSearchCV on the given pipeline and return the best model's feature importance.
    
    Parameters:
    pipeline : sklearn.pipeline.Pipeline
        The machine learning pipeline containing the model.
    xrna_train : pandas.DataFrame
        The training dataset.
    y_train : pandas.Series or numpy array
        The target variable for training.
    param_range : array-like, optional
        The range of values for the 'C' hyperparameter.
    cv_folds : int, optional
        Number of cross-validation folds (default is 5).
    
    Returns:
    tuple
        Best model's feature importance and corresponding feature names.
    """
    #l1/ --LogisticRegression(penalty='l1', and l2 -- LogisticRegression(penalty='elasticnet'
    pipeline = Pipeline([
                    
                    ('model',#LogisticRegression(penalty='l1',solver='saga',random_state=42, max_iter=500, class_weight='balanced',))
                     LogisticRegression(penalty='elasticnet',solver='saga',random_state=42, max_iter=500, class_weight='balanced', l1_ratio=0.8))
                    ])

    search = GridSearchCV(pipeline, {'model__C': param_range}, cv=cv_folds, scoring='roc_auc', verbose=3)
    search.fit(xrna_train, np.ravel(y_train))
    
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    features = list(xrna_train.columns)
    
    selectd=np.array(features)[importance[0] > 0]
    print('Feature importance extracted, number of features:', len(selectd))

    #xrna_train=xrna_train[selectd]

    return selectd


# Argument parser
parser = argparse.ArgumentParser(description='Process multi-omics data with SNF.')
parser.add_argument('--mu_1', type=float, default=0.7, help='Mu parameter for affinity calculation')
parser.add_argument('--snf_k', type=int, default=10, help='Number of neighbors for SNF')
parser.add_argument('--snf_t', type=int, default=10, help='Number of params for SNF') 
parser.add_argument('--snf_metric', type=str, default='sqeuclidean', help='SNF metric') #sqeuclidean
parser.add_argument('--betta', type=float, default=0.9, help='adaptive weight parametr') #weights_mod
parser.add_argument('--alpha', type=float, default=1, help='adaptive weight parametr') #weights_mod

parser.add_argument('--weights_mod', type=float, default=[0.4, 0.3, 0.1, 0.2], help='adaptive weight mod') #weights_mod

parser.add_argument('--select_number', type=int, default=150, help='Number of features to select')
parser.add_argument('--path_data', type=str, default='/Users/sevinjyolchuyeva/Desktop/postdoc/codes/AWSNF/data/afterlasso/', help='Path to input data files')

args = parser.parse_args([])
args_dict = vars(args)
with open("parameters.json", "w") as f:
    json.dump(args_dict, f, indent=4)

print('args:',args_dict)

xrna_train = pd.read_csv(args.path_data + 'xrna_train.csv')
xrna_test = pd.read_csv(args.path_data + 'xrna_test.csv')

xmirna_train = pd.read_csv(args.path_data + 'xmirna_train.csv')
xmirna_test = pd.read_csv(args.path_data + 'xmirna_test.csv')

xcol_train = pd.read_csv(args.path_data + 'xcol_train.csv')
xcol_test = pd.read_csv(args.path_data + 'xcol_test.csv')

xhandc_train = pd.read_csv(args.path_data + 'xhandc_train.csv')
xhandc_test = pd.read_csv(args.path_data + 'xhandc_test.csv')

y_train = pd.read_csv(args.path_data + 'y_train.csv')
y_test = pd.read_csv(args.path_data + 'y_test.csv')

print('Train shape after corr:',xrna_train.shape, xmirna_train.shape, xhandc_train.shape, xcol_train.shape, y_train.shape)

selected_f_col, feature_ranks_col = feature_selection(xcol_train, y_train, 27)
selected_f_handc, feature_ranks_handc = feature_selection(xhandc_train, y_train, args.select_number)
selected_f_mirna, feature_ranks_mirna = feature_selection(xmirna_train, y_train, args.select_number)
selected_f_rna, feature_ranks_rna = feature_selection(xrna_train, y_train, args.select_number)


X_rna_v2=xrna_train[selected_f_rna]
X_mirna_v2=xmirna_train[selected_f_mirna]
X_handc_v2=xhandc_train[selected_f_handc]
X_col_v2=xcol_train.copy()



sorted_weights = process_feature_weights_and_mad(
    X_v2_list=[X_rna_v2, X_mirna_v2,X_handc_v2, X_col_v2],
    feature_ranks_list=[feature_ranks_rna, feature_ranks_mirna,feature_ranks_handc, feature_ranks_col],
    betta=args.betta,
)



X_rna_v2_test=xrna_test[selected_f_rna]
X_mirna_v2_test=xmirna_test[selected_f_mirna]
X_handc_v2_test=xhandc_test[selected_f_handc]


similarity_view1_w = make_affinity_with_weight(X_rna_v2,weight=sorted_weights[0]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view2_w = make_affinity_with_weight(X_mirna_v2,weight=sorted_weights[1]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view3_w = make_affinity_with_weight(X_handc_v2,weight=sorted_weights[2]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view4_w = make_affinity_with_weight(X_col_v2,weight=sorted_weights[3]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)

fused_network = SNF_modality_weights([similarity_view1_w, similarity_view2_w,similarity_view3_w,similarity_view4_w],  K=args.snf_k, t=args.snf_t, weight_modality=args.weights_mod,alpha=args.alpha)

fused_network_test = np.zeros((X_rna_v2_test.shape[0], fused_network.shape[1]))

for i in range(X_rna_v2_test.shape[0]):
    # Append the test sample to the reference data temporarily for similarity calculation
    combined_view1 = np.vstack([X_rna_v2, X_rna_v2_test[i:i+1]])
    combined_view2 = np.vstack([X_mirna_v2, X_mirna_v2_test[i:i+1]])
    combined_view3 = np.vstack([X_handc_v2, X_handc_v2_test[i:i+1]])
    combined_view4 = np.vstack([xcol_train, xcol_test[i:i+1]])
    
    similarity_test1 = make_affinity_with_weight(combined_view1,weight=sorted_weights[0]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test2 = make_affinity_with_weight(combined_view2,weight=sorted_weights[1]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test3 = make_affinity_with_weight(combined_view3,weight=sorted_weights[2]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test4 = make_affinity_with_weight(combined_view4,weight=sorted_weights[3]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)

    # Fuse these affinities for the reference data and the single test sample
    fused_combined = SNF_modality_weights([similarity_test1, similarity_test2,similarity_test3,similarity_test4],  K=args.snf_k, t=args.snf_t, weight_modality=args.weights_mod,alpha=args.alpha)

    # Extract the test sample's similarity to each reference sample

    fused_network_test[i, :] = fused_combined[-1, :-1]  # Exclude self-similarity to get 1x100

print('Fused shape:', fused_network.shape, fused_network_test.shape)
assert (fused_network.shape[1]== fused_network_test.shape[1])



# Define cross-validation strategy
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the base pipeline (classifier will be updated dynamically)
pipe = Pipeline([('classifier', LogisticRegression())])

param_functions = {
    "XGBoost": parameters_xgb(),
    "Random Forest": parameters_randomf(),
    "SVM": parameters_svm(),
}

# Iterate over parameter spaces for different models
for func_name, search_space in param_functions.items():
    print(f'Running GridSearch for: {func_name}')  # Print function name

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=k_fold,
        scoring='roc_auc',
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model
    clf.fit(fused_network, np.ravel(y_train))

    print(f'Best cross-validation score: {clf.best_score_}')
    print(f'Best parameters: {clf.best_params_}')

    # Retrieve the best model
    final_clf = clf.best_estimator_

    # Make predictions
    y_pred = final_clf.predict(fused_network_test)

    # Convert predictions and labels to integers
    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_test.OS_quar_label.to_list()]

    # Compute accuracy
    acc_test = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc_test}')

    auc_score = roc_auc_score(y_true, final_clf.predict_proba(fused_network_test)[:, 1])
    print(f'AUC: {auc_score}')

    # Print confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    if func_name=='SVM':
        y_scores = final_clf.decision_function(fused_network_test)  # No need to select class 1
        auc_score = roc_auc_score(y_test, y_scores)
        print(f"AUC Score (decision function): {auc_score:.4f}")


print('Method for adding weight to the features')


fused_network = snf.snf([similarity_view1_w, similarity_view2_w,similarity_view3_w,similarity_view4_w],  K=args.snf_k, t=args.snf_t,alpha=args.alpha)

fused_network_test = np.zeros((X_rna_v2_test.shape[0], fused_network.shape[1]))

for i in range(X_rna_v2_test.shape[0]):
    # Append the test sample to the reference data temporarily for similarity calculation
    combined_view1 = np.vstack([X_rna_v2, X_rna_v2_test[i:i+1]])
    combined_view2 = np.vstack([X_mirna_v2, X_mirna_v2_test[i:i+1]])
    combined_view3 = np.vstack([X_handc_v2, X_handc_v2_test[i:i+1]])
    combined_view4 = np.vstack([xcol_train, xcol_test[i:i+1]])
    
    similarity_test1 = make_affinity_with_weight(combined_view1,weight=sorted_weights[0]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test2 = make_affinity_with_weight(combined_view2,weight=sorted_weights[1]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test3 = make_affinity_with_weight(combined_view3,weight=sorted_weights[2]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test4 = make_affinity_with_weight(combined_view4,weight=sorted_weights[3]['feature_weight'].to_list(),K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)

    
    # Fuse these affinities for the reference data and the single test sample
    fused_combined = snf.snf([similarity_test1, similarity_test2,similarity_test3,similarity_test4],  K=args.snf_k, t=args.snf_t, alpha=args.alpha)

    # Extract the test sample's similarity to each reference sample

    fused_network_test[i, :] = fused_combined[-1, :-1]  # Exclude self-similarity to get 1x100

print('Fused shape:', fused_network.shape, fused_network_test.shape)
assert (fused_network.shape[1]== fused_network_test.shape[1])


# Define cross-validation strategy
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the base pipeline (classifier will be updated dynamically)
pipe = Pipeline([('classifier', LogisticRegression())])

param_functions = {
    "XGBoost": parameters_xgb(),
    "Random Forest": parameters_randomf(),
    "SVM": parameters_svm(),
}

# Iterate over parameter spaces for different models
for func_name, search_space in param_functions.items():
    print(f'Running GridSearch for: {func_name}')  # Print function name

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=k_fold,
        scoring='roc_auc',
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model
    clf.fit(fused_network, np.ravel(y_train))

    print(f'Best cross-validation score: {clf.best_score_}')
    print(f'Best parameters: {clf.best_params_}')

    # Retrieve the best model
    final_clf = clf.best_estimator_

    # Make predictions
    y_pred = final_clf.predict(fused_network_test)

    # Convert predictions and labels to integers
    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_test.OS_quar_label.to_list()]

    # Compute accuracy
    acc_test = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc_test}')

    auc_score = roc_auc_score(y_true, final_clf.predict_proba(fused_network_test)[:, 1])
    print(f'AUC: {auc_score}')

    # Print confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    if func_name=='SVM':
        y_scores = final_clf.decision_function(fused_network_test)  # No need to select class 1
        auc_score = roc_auc_score(y_test, y_scores)
        print(f"AUC Score (decision function): {auc_score:.4f}")




print('Method adding weight to the modality')

similarity_view1 = snf.make_affinity(X_rna_v2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view2 = snf.make_affinity(X_mirna_v2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view3 = snf.make_affinity(X_handc_v2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view4 = snf.make_affinity(xcol_train,  K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)


fused_network = SNF_modality_weights([similarity_view1, similarity_view2, similarity_view3, similarity_view4], K=args.snf_k, t=args.snf_t,alpha=args.alpha,  weight_modality=args.weights_mod)


fused_network_test = np.zeros((X_rna_v2_test.shape[0], fused_network.shape[1]))

for i in range(X_rna_v2_test.shape[0]):
    # Append the test sample to the reference data temporarily for similarity calculation
    combined_view1 = np.vstack([X_rna_v2, X_rna_v2_test[i:i+1]])
    combined_view2 = np.vstack([X_mirna_v2, X_mirna_v2_test[i:i+1]])
    combined_view3 = np.vstack([X_handc_v2, X_handc_v2_test[i:i+1]])
    combined_view4 = np.vstack([xcol_train, xcol_test[i:i+1]])
    
    similarity_test1 = snf.compute.make_affinity(combined_view1, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test2 = snf.compute.make_affinity(combined_view2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)# euclidean')
    similarity_test3 = snf.compute.make_affinity(combined_view3, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test4 = snf.compute.make_affinity(combined_view4, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)


    # Fuse these affinities for the reference data and the single test sample
    fused_combined = SNF_modality_weights([similarity_test1, similarity_test2,similarity_test3,similarity_test4],  K=args.snf_k, t=args.snf_t, alpha=args.alpha, weight_modality=args.weights_mod)
    
    # Extract the test sample's similarity to each reference sample

    fused_network_test[i, :] = fused_combined[-1, :-1]  # Exclude self-similarity to get 1x100

print('Fused shape:', fused_network.shape, fused_network_test.shape)
assert (fused_network.shape[1]== fused_network_test.shape[1])


# Define cross-validation strategy
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the base pipeline (classifier will be updated dynamically)
pipe = Pipeline([('classifier', LogisticRegression())])

param_functions = {
    "XGBoost": parameters_xgb(),
    "Random Forest": parameters_randomf(),
    "SVM": parameters_svm(),
}

# Iterate over parameter spaces for different models
for func_name, search_space in param_functions.items():
    print(f'Running GridSearch for: {func_name}')  # Print function name

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=k_fold,
        scoring='roc_auc',
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model
    clf.fit(fused_network, np.ravel(y_train))

    print(f'Best cross-validation score: {clf.best_score_}')
    print(f'Best parameters: {clf.best_params_}')

    # Retrieve the best model
    final_clf = clf.best_estimator_

    # Make predictions
    y_pred = final_clf.predict(fused_network_test)

    # Convert predictions and labels to integers
    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_test.OS_quar_label.to_list()]

    # Compute accuracy
    auc_test = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {auc_test}')

    # Print confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))


print('Simple SNF method')

similarity_view1 = snf.make_affinity(X_rna_v2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view2 = snf.make_affinity(X_mirna_v2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view3 = snf.make_affinity(X_handc_v2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
similarity_view4 = snf.make_affinity(xcol_train,  K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)

print('X_rna_v2',X_rna_v2,similarity_view1)
print('xcol_train',xcol_train,similarity_view4)

# Fuse networks
fused_network = snf.snf([similarity_view1, similarity_view2, similarity_view3, similarity_view4], K=args.snf_k, t=args.snf_t,alpha=args.alpha)


fused_network_test = np.zeros((X_rna_v2_test.shape[0], fused_network.shape[1]))

for i in range(X_rna_v2_test.shape[0]):
    # Append the test sample to the reference data temporarily for similarity calculation
    combined_view1 = np.vstack([X_rna_v2, X_rna_v2_test[i:i+1]])
    combined_view2 = np.vstack([X_mirna_v2, X_mirna_v2_test[i:i+1]])
    combined_view3 = np.vstack([X_handc_v2, X_handc_v2_test[i:i+1]])
    combined_view4 = np.vstack([xcol_train, xcol_test[i:i+1]])
    
    similarity_test1 = snf.compute.make_affinity(combined_view1, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test2 = snf.compute.make_affinity(combined_view2, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)# euclidean')
    similarity_test3 = snf.compute.make_affinity(combined_view3, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)
    similarity_test4 = snf.compute.make_affinity(combined_view4, K=args.snf_k, mu=args.mu_1,metric=args.snf_metric,normalize=False)


    # Fuse these affinities for the reference data and the single test sample
    fused_combined = snf.snf([similarity_test1, similarity_test2,similarity_test3,similarity_test4], K=args.snf_k, t=args.snf_t,alpha=args.alpha)
    
    # Extract the test sample's similarity to each reference sample

    fused_network_test[i, :] = fused_combined[-1, :-1]  # Exclude self-similarity to get 1x100

print('Fused shape:', fused_network.shape, fused_network_test.shape)
assert (fused_network.shape[1]== fused_network_test.shape[1])


# Define cross-validation strategy
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the base pipeline (classifier will be updated dynamically)
pipe = Pipeline([('classifier', LogisticRegression())])

param_functions = {
    "XGBoost": parameters_xgb(),
    "Random Forest": parameters_randomf(),
    "SVM": parameters_svm(),
}

# Iterate over parameter spaces for different models
for func_name, search_space in param_functions.items():
    print(f'Running GridSearch for: {func_name}')  # Print function name

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=k_fold,
        scoring='roc_auc',
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model
    clf.fit(fused_network, np.ravel(y_train))

    print(f'Best cross-validation score: {clf.best_score_}')
    print(f'Best parameters: {clf.best_params_}')

    # Retrieve the best model
    final_clf = clf.best_estimator_

    # Make predictions
    y_pred = final_clf.predict(fused_network_test)

    # Convert predictions and labels to integers
    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_test.OS_quar_label.to_list()]

    # Compute accuracy
    auc_test = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {auc_test}')

    # Print confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
