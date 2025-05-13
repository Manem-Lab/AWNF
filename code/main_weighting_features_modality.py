#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from adaptive_weighted_snf import *
from ml_funcs import *
import snf

# ============================
# Feature Processing Functions
# ============================
def drop_highly_correlated(X, target, column_name, threshold=0.9):
    df = pd.concat([X, target], axis=1)
    corr_matrix = df.corr(method='pearson')
    target_corr = corr_matrix[column_name].abs().sort_values(ascending=False)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for column in upper_tri.columns:
        high_corr_features = upper_tri[column][upper_tri[column] > threshold].index.tolist()
        for feature in high_corr_features:
            if target_corr[feature] < target_corr[column]:
                to_drop.add(feature)
            else:
                to_drop.add(column)
    return to_drop

def perform_grid_search(X_train, y_train, param_range=np.arange(0.1, 1.0, 0.1), cv_folds=5):
    pipeline = Pipeline([
        ('model', LogisticRegression(penalty='elasticnet', solver='saga',
                                     random_state=42, max_iter=500, class_weight='balanced', l1_ratio=0.8))
    ])

    search = GridSearchCV(pipeline, {'model__C': param_range}, cv=cv_folds, scoring='roc_auc', verbose=3)
    search.fit(X_train, np.ravel(y_train))
    importance = np.abs(search.best_estimator_.named_steps['model'].coef_)
    selected = np.array(X_train.columns)[importance[0] > 0]
    print('Feature importance extracted, number of features:', len(selected))
    return selected

# ============================
# Argument Parser
# ============================
import argparse

parser = argparse.ArgumentParser(
    description='Run SNF-based model with optional weighting at feature and modality levels.'
)

parser.add_argument('--mu_1', type=float, default=0.7,
                    help='Mu parameter for similarity computation (controls affinity decay).')

parser.add_argument('--snf_k', type=int, default=10,
                    help='Number of neighbors (K) to consider in the SNF algorithm.')

parser.add_argument('--snf_t', type=int, default=10,
                    help='Number of iterations for similarity network fusion.')

parser.add_argument('--snf_metric', type=str, default='sqeuclidean',
                    help='Distance metric for affinity matrix calculation (e.g., sqeuclidean, cosine).')

parser.add_argument('--betta', type=float, default=0.9,
                    help='Beta parameter controlling influence of feature MAD in adaptive weighting.')

parser.add_argument('--alpha', type=float, default=1.0,
                    help='Alpha parameter for SNF fusion (used in normalization).')

parser.add_argument('--weights_mod', type=float, nargs='+', default=[0.4, 0.3, 0.1, 0.2],
                    help='List of weights for each modality during SNF fusion.')

parser.add_argument('--select_number', type=int, default=150,
                    help='Number of top features to select from each modality.')

parser.add_argument('--type_weight', type=str, default='feat_modal',
                    choices=['simple', 'feature', 'modality', 'feat_modal'],
                    help='Strategy for applying weights: "simple" (unweighted), '
                         '"feature" (feature-level), "modality" (modality-level), '
                         'or "feat_modal" (both feature and modality).')


parser.add_argument('--path_data', type=str, required=True)

args = parser.parse_args()
with open("parameters.json", "w") as f:
    json.dump(vars(args), f, indent=4)


# ============================
# Load Data
# ============================
data_path = args.path_data

xrna_train   = pd.read_csv(f"{data_path}xrna_train.csv")
xrna_test    = pd.read_csv(f"{data_path}xrna_test.csv")
xmirna_train = pd.read_csv(f"{data_path}xmirna_train.csv")
xmirna_test  = pd.read_csv(f"{data_path}xmirna_test.csv")
xcol_train   = pd.read_csv(f"{data_path}xcol_train.csv")
xcol_test    = pd.read_csv(f"{data_path}xcol_test.csv")
xhandc_train = pd.read_csv(f"{data_path}xhandc_train.csv")
xhandc_test  = pd.read_csv(f"{data_path}xhandc_test.csv")
y_train      = pd.read_csv(f"{data_path}y_train.csv")
y_test       = pd.read_csv(f"{data_path}y_test.csv")

print('Train data shapes:', xrna_train.shape, xmirna_train.shape, xhandc_train.shape, xcol_train.shape, y_train.shape)

# ============================
# Feature Selection
# ============================
selected_rna,   ranks_rna   = feature_selection(xrna_train, y_train, args.select_number)
selected_mirna, ranks_mirna = feature_selection(xmirna_train, y_train, args.select_number)
selected_handc, ranks_handc = feature_selection(xhandc_train, y_train, args.select_number)
selected_col,   ranks_col   = feature_selection(xcol_train, y_train, 27)

X_rna_v2     = xrna_train[selected_rna]
X_mirna_v2   = xmirna_train[selected_mirna]
X_handc_v2   = xhandc_train[selected_handc]
X_col_v2     = xcol_train.copy()

X_rna_v2_test   = xrna_test[selected_rna]
X_mirna_v2_test = xmirna_test[selected_mirna]
X_handc_v2_test = xhandc_test[selected_handc]

X_train_views = [X_rna_v2, X_mirna_v2, X_handc_v2, X_col_v2]
X_test_views  = [X_rna_v2_test, X_mirna_v2_test, X_handc_v2_test, xcol_test]

# ============================
# Feature Weights (if needed)
# ============================
if args.type_weight in ['feature', 'feat_modal']:
    sorted_weights = process_feature_weights_and_mad(
        X_v2_list=X_train_views,
        feature_ranks_list=[ranks_rna, ranks_mirna, ranks_handc, ranks_col],
        betta=args.betta
    )

# ============================
# SNF Fusion (Train)
# ============================
if args.type_weight == 'feat_modal':
    similarity_views = [
        make_affinity_with_weight(view, weight=sorted_weights[i]['feature_weight'].tolist(),
                                  K=args.snf_k, mu=args.mu_1, metric=args.snf_metric, normalize=False)
        for i, view in enumerate(X_train_views)
    ]
    fused_network = SNF_modality_weights(similarity_views, K=args.snf_k, t=args.snf_t,
                                         weight_modality=args.weights_mod, alpha=args.alpha)

elif args.type_weight == 'feature':
    similarity_views = [
        make_affinity_with_weight(view, weight=sorted_weights[i]['feature_weight'].tolist(),
                                  K=args.snf_k, mu=args.mu_1, metric=args.snf_metric, normalize=False)
        for i, view in enumerate(X_train_views)
    ]
    fused_network = snf.snf(similarity_views, K=args.snf_k, t=args.snf_t, alpha=args.alpha)

elif args.type_weight == 'simple':
    similarity_views = [
        snf.make_affinity(view, K=args.snf_k, mu=args.mu_1, metric=args.snf_metric, normalize=False)
        for view in X_train_views
    ]
    fused_network = snf.snf(similarity_views, K=args.snf_k, t=args.snf_t, alpha=args.alpha)

elif args.type_weight == 'modality':
    similarity_views = [
        snf.make_affinity(view, K=args.snf_k, mu=args.mu_1, metric=args.snf_metric, normalize=False)
        for view in X_train_views
    ]
    fused_network = SNF_modality_weights(similarity_views, K=args.snf_k, t=args.snf_t,
                                         weight_modality=args.weights_mod, alpha=args.alpha)

else:
    raise ValueError(f"Invalid type_weight: {args.type_weight}")

# ============================
# SNF Fusion (Test)
# ============================
fused_test = compute_fused_test_network(
    X_train_views=X_train_views,
    X_test_views=X_test_views,
    sorted_weights=sorted_weights if args.type_weight in ['feature', 'feat_modal'] else None,
    snf_k=args.snf_k,
    snf_t=args.snf_t,
    mu_1=args.mu_1,
    metric=args.snf_metric,
    weight_modality=args.weights_mod,
    alpha=args.alpha,
    type_weight=args.type_weight
)

assert fused_network.shape[1] == fused_test.shape[1]
print("Fusion complete. Output shape:", fused_test.shape)


# ============================
# Model Training & Evaluation
# ============================
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([('classifier', LogisticRegression())])

models = {
    "XGBoost": parameters_xgb(),
    "Random Forest": parameters_randomf(),
}

for model_name, param_grid in models.items():
    print(f'Running GridSearch for: {model_name}')
    clf = GridSearchCV(pipe, param_grid, cv=k_fold, scoring='roc_auc', refit=True, verbose=1, n_jobs=-1)
    clf.fit(fused_network, np.ravel(y_train))
    final_model = clf.best_estimator_

    y_pred = final_model.predict(fused_test)
    y_true = y_test.OS_quar_label.astype(int).tolist()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, final_model.predict_proba(fused_test)[:, 1])

    print(f"Model: {model_name}")
    print(f"Best CV Score: {clf.best_score_}")
    print(f"Best Parameters: {clf.best_params_}")
    print(f"Test Accuracy: {acc}")
    print(f"Test AUC: {auc}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

