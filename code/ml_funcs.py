from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import uniform


def parameters_xgb():
    search_space = [{
        'classifier': [XGBClassifier(
            random_state=42, seed=42, booster='gbtree',
            use_label_encoder=False, eval_metric='auc'
        )],
        'classifier__n_estimators': [100, 200, 300, 400],#500],
        #'classifier__reg_lambda': [0, 0.1, 0.2, 0.25],
        'classifier__eta': [0.1, 0.2, 1,15,0.3],
        'classifier__subsample': [0.9, 0.8],
        # 'classifier__max_depth': [2, 4, 6]
    }]
    return search_space

def parameters_svm():
    search_space = [{
        'classifier': [SVC(random_state=42, probability=True)],
        'classifier__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
        'classifier__kernel': ["rbf", "sigmoid",], #"linear"],
        #'classifier__gamma': ['scale', 'auto'] + list(uniform(0.001, 0.999).rvs(10)),
        'classifier__gamma': ['scale', 'auto'],

        #'classifier__degree': [1, 2,3]
    }]
    return search_space

def parameters_randomf():
    search_space = [{
        'classifier': [RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight='balanced'
        )],
        'classifier__n_estimators': [50, 100, 200, 300,350],# 400],
        #'classifier__max_depth': [3, 5]
    }]
    return search_space
