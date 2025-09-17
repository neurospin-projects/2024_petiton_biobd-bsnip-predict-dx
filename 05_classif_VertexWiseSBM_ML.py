from utils import get_predict_sites_list, get_classifier, get_scores, save_pkl, read_pkl

import numpy as np, gc, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import sys, os
from tqdm import tqdm
from deep_learning_sbm.data_transforms import create_initial_transforms
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer



MODALITIES_SBM_VERTEXWISE = ["surface-lh_data", "surface-rh_data"]
DATA_DIR_SBM_VERTEXWISE= "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
SPLITS_DICT_SBM_VERTEXWISE="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/dict_splits_test_all_trainset_sizes_N763.pkl"

# outputs
RESULTSFOLDER="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifSBM/"

def get_all_data(verbose=False):
    data_dict = {}
    metadata = pd.read_csv(DATA_DIR_SBM_VERTEXWISE + "metadata.tsv", sep="\t")
    for mod in MODALITIES_SBM_VERTEXWISE:
        data_path = os.path.join(DATA_DIR_SBM_VERTEXWISE, f"{mod}.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading {data_path}...")
        all_data = np.load(data_path)  # Shape: (n_subjects, n_vertices, n_metrics) =  (763, 163842, 3)
        data_dict[mod] = all_data
        print(f"Loaded {mod}: {all_data.shape}")

    # apply transforms to data, otherwise the nb of parameters is too huge
    data = {}
    initial_transform = create_initial_transforms()
    for mod, d in data_dict.items():
        if verbose: print(f"Applying initial transform to {mod}: {d.shape} -> ", end="") # shape now is (763, 3, 10242)
        # Apply initial transform to full data, then subset
        transformed_data = initial_transform(d)  # (n_subjects, n_metrics, n_vertices_downsampled)
        data[mod] = transformed_data  
        if verbose: print(f"{data[mod].shape}")

    # concatenate left and right hemispheres (763, 3, 20484)
    big_array_lh_rh=np.concatenate([data[MODALITIES_SBM_VERTEXWISE[0]], data[MODALITIES_SBM_VERTEXWISE[1]]], axis=2)
    n_subjects, n_metrics, n_vertices_downsampled = np.shape(big_array_lh_rh) 
    # flatten to get all metrics ("thickness", "curv", "sulc") in one dimension
    big_array_lh_rh_flat = big_array_lh_rh.reshape(n_subjects, n_metrics*n_vertices_downsampled)
    if verbose: print(big_array_lh_rh_flat.shape) # (763, 61452) for max train set size

    # create dataframe to have the participant_id corresponding to each numpy array of 3D VBM MRI values
    data = pd.DataFrame({
        'participant_id': metadata['participant_id'], 
        'data': [np.array(row) for row in big_array_lh_rh_flat] 
    })
    print(data)

    return data, metadata, big_array_lh_rh_flat

def load_images(site, data_arr, metadata, datasize = 700, verbose=False):
    """
        site : (str) LOSO-CV site/fold. must be "Baltimore", "Boston", "Dallas",
                 "Detroit", "Hartford", "mannheim", "creteil", "udine", "galway", "pittsburgh", "grenoble", or "geneve".
        data : (df) pandas dataframe with 2 columns, "participant_id" and "data", where each row contains
                the participant_id and its corresponding 3D VBM MRI numpy array 
        datasize : (int) approximate size of the training dataset (it varies depending on the LOSO-CV site as the test set is fixed)
                must be a value in list [100,175,250,350,400,500,600,700,800]

    """
    
    assert site in get_predict_sites_list()," wrong LOSO-CV site name!"
    list_datasizes = [75, 150, 200, 300, 400, 450, 500, 600, 700]
    assert datasize in list_datasizes, "wrong training data size!"

    if verbose: print("np.shape metadata",np.shape(metadata))
    split_data = read_pkl(SPLITS_DICT_SBM_VERTEXWISE)
    traindata_size_idx = list_datasizes.index(datasize)
    if verbose : print("traindata_size ",traindata_size_idx)
    split_key = site + "_" + str(traindata_size_idx)
    indices_tr, indices_te = split_data[split_key]["train"], split_data[split_key]["test"]
    metadata_tr = metadata.iloc[indices_tr].reset_index(drop=True)
    metadata_te = metadata.iloc[indices_te].reset_index(drop=True)

    participant_id_tr = metadata_tr["participant_id"].values
    participant_id_te = metadata_te["participant_id"].values
    
    y_train = metadata_tr["dx"].values.astype(int)
    y_test = metadata_te["dx"].values.astype(int)

    all_labels = metadata.iloc[indices_tr+indices_te]["dx"].values.astype(int)
    if verbose:
        print(f"Dataset initialized with {len(all_labels)} samples")
        print(f"Label distribution: {np.bincount(all_labels)}") 

    X_train = data_arr[indices_tr]
    X_test = data_arr[indices_te]

    if verbose: 
        print("X_train",np.shape(X_train), type(X_train))
        print("X_test",np.shape(X_test), type(X_test))

    # prepare residualizer for residualization on age, sex, and site 
    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    if datasize==700: assert len(metadata)==len(indices_tr)+len(indices_te)
    residualizer = Residualizer(data=metadata, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(metadata)

    Zres_tr = Zres[indices_tr]
    Zres_te = Zres[indices_te]
                     
    gc.collect()

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, \
            "Zres_tr":Zres_tr, "Zres_te": Zres_te, "residualizer":residualizer, "participant_id_tr":participant_id_tr, \
                "participant_id_te":participant_id_te}


def classif_vertexwise_SBM(classif="svm", datasize=700, save=False):
    assert classif in ["svm", "MLP", "EN", "L2LR", "xgboost"]
    data, metadata, data_arr = get_all_data()
    
    assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700], "wrong training dataset size!"
    
    print("training dataset approximate size :", datasize)

    roc_auc_list, bacc_list = [],[]
    metrics_dict = dict()

    for site in tqdm(get_predict_sites_list(), desc=f"Processing sites ({classif})"): 

        # load images for current site and training dataset size
        dict_variables = load_images(site, data_arr, metadata, datasize)
        X_train, X_test = dict_variables["X_train"], dict_variables["X_test"]
        y_train, y_test = dict_variables["y_train"], dict_variables["y_test"]
        Zres_tr, Zres_te = dict_variables["Zres_tr"], dict_variables["Zres_te"]
        ids_tr, ids_te = dict_variables["participant_id_tr"], dict_variables["participant_id_te"]
        residualizer = dict_variables["residualizer"]
        
        print(" site : ", site)
       
        # print("Xtrain",np.shape(X_train), "Xtest",np.shape(X_test))
        # print("Zres_train",np.shape(Zres_tr), "Zres_test",np.shape(Zres_te))
        # print("y_train",np.shape(y_train), "y_test",np.shape(y_test))

        # get classifier
        classifier = get_classifier(classif)

        # fit residualizer
        residualizer.fit(X_train, Zres_tr)
        X_train = residualizer.transform(X_train, Zres_tr)
        X_test = residualizer.transform(X_test, Zres_te)

        # fit scaler
        scaler_ = StandardScaler()
        X_train = scaler_.fit_transform(X_train)
        X_test = scaler_.transform(X_test)

        classifier.fit(X_train, y_train)
        # classifier.fit(tr_,np.random.permutation(y_train))
        y_pred = classifier.predict(X_test)
        y_pred_tr = classifier.predict(X_train)

        # get classification scores for current classifier
        score_test = get_scores(classifier, X_test)  
        score_train = get_scores(classifier, X_train)        
        
        bacc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, score_test)
        roc_auc_list.append(roc_auc)
        bacc_list.append(bacc)

        metrics_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc, "y_pred_te":y_pred, "y_pred_tr": y_pred_tr, "score_test":score_test,\
                              "score_train":score_train, "y_train":y_train, "y_test" :y_test, "participant_ids_tr":ids_tr, "participant_ids_te":ids_te}

        print(site, " roc auc : ", round(100*roc_auc,2),"  balanced accuracy :", round(100*bacc,2))

    print("roc auc moyenne entre tous les sites : ",round(np.mean(roc_auc_list)*100,3))
    print("balanced accuracy moyenne entre tous les sites : ",round(np.mean(bacc_list)*100,3))

    metrics_dict["mean over all sites"] = {"roc_auc": np.mean(roc_auc_list), "balanced-accuracy":np.mean(bacc_list)}

    cpt =0
    print("classification model : ", classif)
    for site_ in get_predict_sites_list():
        print("site ",site_, " roc_auc ",np.round(100*roc_auc_list[cpt],3), " balanced accuracy :",np.round(100*bacc_list[cpt],3))
        cpt+=1
    
    if save:
        results_file = RESULTSFOLDER+classif+"_N"+str(datasize)+"_SBM_vertexwise_N763.pkl"
        print("\nsaving classification results ...")
        save_pkl(metrics_dict, results_file)


def main():    
    for size in [75, 150, 200, 300, 400, 450, 500, 600]: #75, 150, 200, 300, 400
        for classifier in ["svm"]:#"svm", "MLP", "EN", "L2LR"]:#, "xgboost"]:
            print(classifier)
            classif_vertexwise_SBM(classif=classifier, datasize=size, save=True)
        
    # parser = argparse.ArgumentParser()

    # # choose the size of the training set (Nmax=800) for classification 
    # parser.add_argument("--datasize", type=int, choices = [75, 150, 200, 300, 400, 450, 500, 600, 700])
    # # choose the classifier model
    # parser.add_argument("--model", type=str, choices = ["svm", "MLP","EN","xgboost","L2LR"])

    # keyboard_args = parser.parse_args()

    # classif_voxelwise_VBM(keyboard_args.model, datasize = keyboard_args.datasize)

if __name__ == '__main__':
    main()
