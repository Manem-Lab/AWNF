# Adaptive Integration of Heterogeneous Biomedical Data in Patient Networks

[[Paper & Supplementary]]

## Abstract
**Background**: Integrating complex patient data, including multi-omics, clinical, and imaging information, remains challenging. Traditional Patient Similarity Networks (PSNs) and existing network fusion methods rely on static similarity measures and often fail to account for varying modality relevance, limiting their ability to capture biologically meaningful patient relationships. 
**Methods**: We propose Adaptive Weighted Network Fusion (AWNF), a novel framework that extends traditional SNF by incorporating feature- and modality-level weighting. This dual weighting strategy dynamically prioritizes the most informative features and modalities, ensuring proportional contribution from each data type and feature. AWNF was evaluated with an XGBoost classifier across four TCGA cancer cohorts, BRCA, LUAD, LUSC, and PRAD, integrating mRNA, miRNA, and DNA methylation to construct a comprehensive multi-omics representation. 
**Results**: AWNF consistently outperformed traditional SNF across all four cancer types, achieving improvements in AUC (~3-8%), F1-score (2-6%), and accuracy (2-5%), with the largest gains observed in LUAD and PRAD. These results demonstrate that adaptively weighting features and modalities allows AWNF to better capture patient similarities and improve downstream predictive performance across diverse cancers. 
**Conclusion**: AWNF represents a substantial advance over existing network fusion methods, providing a scalable, interpretable, and biologically meaningful framework for multimodal patient network construction. Its adaptive weighting enables robust integration of heterogeneous data, improving modeling of cancer heterogeneity and predictive accuracy across multiple cancer types.


# AWNF Package

AWNF is a Python package designed to implement **Similarity Network Fusion (SNF)** with an **adaptive weighting mechanism**. This package is intended for use in computational biology, machine learning, and data science tasks that involve multi-view data, such as genomics, imaging, and clinical data.

The package helps integrate heterogeneous data sources into a single, unified similarity network, which can be used for predictive modelling and analysis.

---

## Features

- **Adaptive Weighted Network Fusion (AWNF)**: A flexible framework for integrating multi-modality.


---

## Installation

To install the **weighted_snf** package, you can use **pip**.

### Install via PyPI (if published)
```bash
pip install awnf
```

## Usage

Once installed, you can use the **awnf** package by importing its functions into your Python scripts. Below is an example usage:

### Example
```python

# Import necessary functions from the 'awnf' package and other libraries
from awnf import feature_selection, make_affinity_with_weight, SNF_modality_weights, process_feature_weights_and_mad
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Example input data for affinity matrix calculation
# First, generate a synthetic dataset to demonstrate feature selection and affinity matrix calculation

# Generate a classification dataset with 100 samples, 50 features, and 2 informative features
# The dataset will have 2 classes (target variable)
X, y = make_classification(n_samples=100, n_features=50, n_informative=2, n_classes=2, random_state=42)

# Convert the data into a pandas DataFrame for easier manipulation and inspection
X_df1 = pd.DataFrame(X, columns=[f'Feat_mod1_{i+1}' for i in range(X.shape[1])])
y_df1 = pd.DataFrame(y, columns=['Target'])

# Generate another classification dataset with a different set of features
X, y = make_classification(n_samples=100, n_features=60, n_informative=2, n_classes=2, random_state=42)

# Convert the second dataset to a pandas DataFrame for ease of manipulation
X_df2 = pd.DataFrame(X, columns=[f'Feat_mod2_{i+1}' for i in range(X.shape[1])])
y_df2 = pd.DataFrame(y, columns=['Target'])

# Display the first few rows of the first dummy dataset (X_df1 and y_df1)
print(X_df1.head())
print(y_df1.head())

# Perform feature selection using the Boruta algorithm for the first dataset (X_df1)
num_features = 10  # Specify the number of features to select
selected_genes1, feature_ranks1 = feature_selection(X_df1, np.ravel(y_df1), num_features=num_features, n_estimators=100)

# Display the feature ranks for the first dataset
print(feature_ranks1)

# Perform feature selection on the second dataset (X_df2)
num_features = 18  # Specify the number of features to select
selected_genes2, feature_ranks2 = feature_selection(X_df2, np.ravel(y_df2), num_features=num_features, n_estimators=50)

# Display the feature ranks for the second dataset
print(feature_ranks2)

# Update the feature sets for both datasets by selecting only the top-ranked features
X_df1 = X_df1[selected_genes1]
X_df2 = X_df2[selected_genes2]

# Process the feature weights and calculate the feature importance
# The 'process_feature_weights_and_mad' function is assumed to calculate weights based on feature ranks
sorted_weights = process_feature_weights_and_mad(
    X_v2_list=[X_df1, X_df2],  # List of feature datasets
    feature_ranks_list=[feature_ranks1, feature_ranks2],  # Corresponding feature ranks
    betta=0.5,  # A parameter controlling the weight scaling (assumed)
)

# Display the first set of weights (for the first dataset)
print(sorted_weights[0])

# Generate the similarity (affinity) matrices for each dataset using the feature weights
similarity_view1_w = make_affinity_with_weight(X_df1, weight=sorted_weights[0]['feature_weight'].to_list())
similarity_view2_w = make_affinity_with_weight(X_df2, weight=sorted_weights[1]['feature_weight'].to_list())

# Combine the similarity matrices from both datasets using AWNF (Similarity Network Fusion)
# We are assigning different weights to the modalities (views) based on their importance
fused_network = SNF_modality_weights([similarity_view1_w, similarity_view2_w], weight_modality=[0.8, 0.2])

# Print the resulting fused network, which combines information from both datasets
print('fused_network', fused_network)

# ----------------------------
# Compute Fused Similarity for Test Data
# Example only: you must define X_train_views, X_test_views, args beforehand

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
```

## 📘 Notes

- `feature_selection()`: Performs Boruta-based selection on individual modalities.
- `make_affinity_with_weight()`: Computes weighted affinity matrices based on feature importance.
- `process_feature_weights_and_mad()`: Calculates weights using feature rank and dispersion.
- `SNF_modality_weights()`: Fuses similarity matrices with modality-level weighting.
- `compute_fused_test_network()`: Predicts similarity between test and training samples across fused networks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

AWNF is developed based on the following code repositories:
1. [SNFpy](https://github.com/rmarkello/snfpy)
2. [boruta_py](https://github.com/scikit-learn-contrib/boruta_py)


We are very grateful for their contributions to the community.

## Authors

**Sevinj Yolchuyeva, Venkata Manem**  
Email: [sevinj.yolchuyeva@crchudequebec.ulaval.ca](mailto:sevinj.yolchuyeva@crchudequebec.ulaval.ca), [venkata.manem@crchudequebec.ulaval.ca](mailto:venkata.manem@crchudequebec.ulaval.ca)

Centre de Recherche du CHU de Québec - Université Laval
