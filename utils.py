import os , pandas as pd, numpy as np, json, gc, sys
from sklearn import svm
import pickle, shap
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm
from xgboost import XGBClassifier 
import pandas as pd
import nibabel
from torchvision.transforms.transforms import Compose
from transforms import Crop, Padding, Normalize
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster


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

def get_scores_pipeline(pipeline, X_test_res, X_train_res=None):
    try:
        score_test = pipeline.predict_proba(X_test_res)[:, 1]
        if X_train_res: score_train = pipeline.predict_proba(X_train_res)[:, 1]
    except AttributeError:
        score_test = score_train = None
        try:
            score_test = pipeline.predict_log_proba(X_test_res)[:, 1]
            if X_train_res : score_train = pipeline.predict_log_proba(X_train_res)[:, 1]
        except AttributeError:
            try:
                score_test = pipeline.decision_function(X_test_res)
                if X_train_res: score_train = pipeline.decision_function(X_train_res)
            except AttributeError:
                raise RuntimeError("Classifier does not implement a supported scoring method.")

    if X_train_res : return score_test, score_train
    else : return score_test

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

def save_shap_file(shap_values, shapfile):
    file_name = f"{shapfile}.pkl"
    if not os.path.exists(file_name):
        save_pkl(shap_values, file_name)
        print(f"Saved: {file_name}")
        return  # Stop once we save a file

def compute_covariance(X, y):
    # Number of samples (n) and features (m)
    n, m = X.shape

    # Mean of the scores_test
    y_bar = np.mean(y)
    
    # Mean of each feature in X
    Xj_bar = np.mean(X, axis=0)
    print(f"Xj_bar {Xj_bar.shape}")
    
    # Center the features and scores
    X_centered = X - Xj_bar
    y_centered = y - y_bar
    print(f"X_centered {X_centered.shape}")
    print(f"y_centered {y_centered.shape}")

    # Compute covariance between each feature and scores_test
    covariance = (X_centered.T @ y_centered) / (n - 1)
    return covariance

def get_reshaped_4D(array, brain_mask_path_):
    nifti_mask = nibabel.load(brain_mask_path_)
    nifti_data_mask = nifti_mask.get_fdata()
    image_shape = nifti_data_mask.shape
    # flatten mask
    print("mask shape in 3D", np.shape(nifti_data_mask))
    nifti_data_mask = nifti_data_mask.ravel()
    # all reshaped zscores for this path
    nb_subjects = array.shape[0]
    print("nb_subjects  ",nb_subjects)
    # on copie nb_subjects fois le masque de sorte à avoir un array de taille (nb_subjects,taille_masque_avec_0_et_1)
    result_array_ = np.tile(nifti_data_mask, (nb_subjects, 1))
    # where each subject of result_array is equal to 1, we copy the contents of the baltimore subjects 
    print("result_array_.shape[0]",result_array_.shape[0])
    for subject in range(0,result_array_.shape[0]): # loop through subjects
        result_array_[subject,nifti_data_mask==1] = array[subject,:]
    reshaped_img = result_array_.reshape((nb_subjects,*image_shape))

    return reshaped_img

def inverse_transform(arr):
    """
    Inverse the transformations applied by cropping and padding.

    Parameters:
        arr (np.ndarray): Input array with squeezed shapes (121, 128, 121), or (128, 128, 128).

    Returns:
        np.ndarray: Array after inverse transformations, restored to the shape (121, 145, 121).
    """
    # Ensure the array shape is as expected
    assert arr.ndim == 3, f"Expected a 3D array, but got shape {arr.shape}"

    # Step 1: Undo Padding
    if arr.shape == (128, 128, 128):  # From padded (1, 128, 128, 128)
        arr = arr[3:124, 4:125, 4:125]  # Crop to (121, 121, 121)

    # Step 2: Undo Cropping
    # Restore to (121, 145, 121) by padding
    pad_x = (145 - arr.shape[1]) // 2  # Padding for x-axis
    pad_y = (121 - arr.shape[2]) // 2  # Padding for y-axis

    arr = np.pad(
        arr,
        ((0, 0), (pad_x, 145 - arr.shape[1] - pad_x), (pad_y, 121 - arr.shape[2] - pad_y)),
        mode="constant",
    )

    return arr

# Function to round to 2 significant digits and keep scientific notation
def round_sci(x, sig=2):
    if pd.isna(x) or x == 0:
            return str(x)
    return f"{x:.{sig}e}".replace("e+0", "e").replace("e+","e").replace("e0", "e")


def get_threshold_for_k_clusters(linkage_matrix, k):
    for t in np.linspace(0, linkage_matrix[:, 2].max(), 500):
        labels = fcluster(linkage_matrix, t=t, criterion='distance')
        if len(np.unique(labels)) == k:
            return t
    return None


def plot_dendrogram(model, nb_clusters=6):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    #This creates a linkage matrix in the format required by scipy.cluster.hierarchy.dendrogram, where :
    # each row: [idx1, idx2, distance, count]
    # idx1 and idx2: clusters merged
    # distance: the distance between them (from model.distances_)
    # count: how many original features are in the newly formed cluster

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
    threshold = get_threshold_for_k_clusters(linkage_matrix, k=nb_clusters)


    # Plot the corresponding dendrogram
    dendro = dendrogram(linkage_matrix, color_threshold=threshold, above_threshold_color='gray')
    plt.grid(False)
    plt.xticks(rotation=35, ha='right', fontsize=20) 
    plt.subplots_adjust(bottom=0.3)
    plt.tick_params(axis='both', labelsize=10) 


    return dendro


def get_gm_csf_index_map(names):
    """
    From a list of ROI names ending in _GM_Vol and _CSF_Vol,
    return a dict mapping {ROI_name: (GM_index, CSF_index)}.
    """
    pairs = {}
    for i, name in enumerate(names):
        base = name.rsplit("_", 2)[0]  # remove _GM_Vol or _CSF_Vol
        if base not in pairs:
            pairs[base] = [None, None]  # [GM_index, CSF_index]
        
        if "_GM_Vol" in name:
            pairs[base][0] = i
        elif "_CSF_Vol" in name:
            pairs[base][1] = i
    
    return {k: tuple(v) for k, v in pairs.items()}

def get_lr_index_map(names):
    """
    From a list of region names where left hemisphere names start with 'l'
    and right hemisphere names start with 'r', return:
    {region_name_without_prefix: (index_left, index_right)}
    Only includes regions that have both left and right versions.
    """
    lr_dict = {}
    
    name_to_index = {name: i for i, name in enumerate(names)}
    
    for name in names:
        if name.startswith('l'):
            base = name[1:]  # remove 'l'
            right_name = 'r' + base
            if right_name in name_to_index:
                lr_dict[base] = (name_to_index[name], name_to_index[right_name])
    
    return lr_dict