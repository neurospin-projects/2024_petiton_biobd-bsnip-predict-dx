import os , pandas as pd, numpy as np, json, gc, sys
from sklearn import svm
import pickle, shap
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm
from xgboost import XGBClassifier 
import pandas as pd


DATAFOLDER="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"

def get_LOSO_CV_splits_N763():
    with open(DATAFOLDER+"splitsBD_LOSOCV_SBM_ROI.pkl", "rb") as f:
        splits = pickle.load(f)
    return splits

def get_LOSO_CV_splits_N861():
    with open(DATAFOLDER+"splitsBD_LOSOCV_VBM_ROI.pkl", "rb") as f:
        splits = pickle.load(f)
    return splits

def get_classifier(classif):
    if classif =="svm" :
        classifier= GridSearchCV(estimator = svm.SVC(class_weight='balanced'), param_grid={'kernel': ['rbf'], 'gamma' : ["scale"],'C': [ 0.1,  1. , 10. ]},\
                                cv=5, n_jobs=1)
    if classif =="EN":
        classifier = GridSearchCV(estimator=lm.SGDClassifier(loss='log_loss', penalty='elasticnet',class_weight='balanced',random_state=42),
                                                    param_grid={'alpha': 10. ** np.arange(-1, 2),
                                                                'l1_ratio': [.1, .5, .9]},
                                                                cv=3, n_jobs=1)
    if classif=="L2LR" : 
        classifier = lm.LogisticRegression(C=10.0, class_weight='balanced', fit_intercept=False)

    if classif == "xgboost":
        classifier = GridSearchCV(estimator=XGBClassifier(random_state=0, n_jobs=1),
                                param_grid={"n_estimators": [10, 50, 100],
                                            "learning_rate":[0.05, 0.1],
                                            "max_depth":[3, 6],
                                            "subsample":[0.8]}, cv=3, n_jobs=1)
    if classif =="MLP":
        mlp_param_grid = {"hidden_layer_sizes":
                    [(100, ), (50, ), (25, ), (10, ), (5, ),          # 1 hidden layer
                    (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),  # 2 hidden layers
                    (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )], # 3 hidden layers
                    "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}
        
        classifier = GridSearchCV(estimator = MLPClassifier(random_state=1), param_grid=mlp_param_grid, cv=3, n_jobs=1)
    return classifier


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

def get_scores(classifier, X_test):
    if hasattr(classifier, 'decision_function'):
        print("decision_function")
        score_test = classifier.decision_function(X_test)
    elif hasattr(classifier, 'predict_log_proba'):
        print("log proba")
        score_test = classifier.predict_log_proba(X_test)[:, 1]
    elif hasattr(classifier, 'predict_proba'):
        print("predict proba")
        score_test = classifier.predict_proba(X_test)[:, 1]
    return score_test

def has_nan(arr):
    # if there are nan values in the array
    return np.isnan(arr).any()
def has_inf(arr):
    # if there are nan values in the array
    return np.isinf(arr).any()
def has_zero(arr):
    # if there are zero values in the array
    return np.any(arr == 0)
    
def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

def save_pkl(dict_or_array, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_or_array, file)
    print(f'Item saved to {file_path}')

def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_participants():
    return pd.read_csv(DATAFOLDER+"participantsBD.csv")

def get_predict_sites_list():
    return ["Baltimore", "Boston", "Dallas", "Detroit", "Hartford", "mannheim", "creteil", "udine", "galway", "pittsburgh", "grenoble", "geneve"]
 

def has_zeros_col(arr):
    # if there are columns with only values equal to zero in the array
    zero_columns = np.all(arr == 0, axis=0)
    return np.any(zero_columns)
