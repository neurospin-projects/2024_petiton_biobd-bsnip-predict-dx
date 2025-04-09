import os, pickle, json, gc, sys, numpy as np, pandas as pd, shap, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from tqdm import tqdm
from utils import save_pkl, get_participants, get_predict_sites_list, get_classifier,\
        get_LOSO_CV_splits_N861, get_LOSO_CV_splits_N763, get_scores, create_folder_if_not_exists, save_shap_file

import time
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

#inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATAFOLDER=ROOT+"data/processed/"

#outouts
RESULTSFOLDER=ROOT+"results_classif/"
SHAP_DIR=ROOT+"models/ShapValues/shap_computed_from_all_Xtrain/"

def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))


def remove_zeros(df, verbose=False):
    columns_with_only_zeros = df.columns[(df == 0).all()]
    if verbose : print("columns with zeros ",columns_with_only_zeros)
    column_indices = [df.columns.get_loc(col) for col in columns_with_only_zeros]
    assert set(columns_with_only_zeros) <= {'lInfLatVen_GM_Vol', 'lOC_GM_Vol', 'lInfLatVen_CSF_Vol', 'lOC_CSF_Vol'}
    df = df.drop(columns=columns_with_only_zeros)
    return df



def classif_vbm_ROI(classif='svm', datasize=800, save = False, N763=False, compute_shap=False, atlas="Neuromorphometrics", \
                    random_labels=False, onesite=None):
    """ 
        classif : (str) classifier name (it has to be "MLP","L2LR","svm","xgboost", or "EN")
        datasize : (int) has to be 75, 150, 200, 300, 400, 450, 500, 600, or 700 if N763 is true, otherwise
                    the value has to be 100, 175, 250, 350, 400, 500, 600, 700, or 800
        save : (bool) whether we save the pkl file containing roc auc and balanced accuracy measures for each LOSO-CV site
        N763: (bool) whether we classify all subjects (max training set size would then be Nmax = 861)
                    or whether we classify only subjects that have both SBM ROI and VBM ROI available (763 subjects)
        compute_shap: (bool) wether we compute shap values. only applied for maximum training set size and the best-performing classifier
                for VBM ROI, which is a support vector machine with a radial basis function kernel ("svm" for "classif" variable)
        atlas : (str) "Neuromorphometrics", the atlas name used to generate VBM ROI 
        random_labels : (bool) use random labels for training and testing to compute shap with random labels
        onesite : (None or str) to compute Shap one site at a time (onesite is the name of said site) to compute shap 
                for each site in parallel if needed
    """
    if onesite is not None: assert onesite in get_predict_sites_list()
    if random_labels: randomizedlabels_str="_randomized_labels"
    else: randomizedlabels_str=""

    assert classif in ["svm", "MLP", "EN", "L2LR", "xgboost"]
    # read splits
    if N763: 
        splits = get_LOSO_CV_splits_N763()    
        assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"
    else: 
        splits = get_LOSO_CV_splits_N861()    
        assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"

    # read participants dataframe
    participants = get_participants()
     # prepare residualizer for residualization on age, sex, and site 
    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    # select the participants for VBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    if N763: Nmax = 700
    else: Nmax = 800
    participants_all = list(splits["Baltimore-"+str(Nmax)][0])+list(splits["Baltimore-"+str(Nmax)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_VBM = participants.iloc[msk]   
    participants_VBM = participants_VBM.reset_index(drop=True)

    residualizer = Residualizer(data=participants_VBM, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants_VBM)

    dict_score_by_site = dict()
    
    VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    # reorder VBMdf to have rows in the same order as participants_VBM
    VBMdf = VBMdf.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()
    
    roc_auc_list ,scores_list, bacc_list = [],[], []
    metrics_dict , dict_score_by_site = {}, {}

    if onesite is not None: list_sites = [onesite]
    else : list_sites = get_predict_sites_list()

    # loop through LOSO-CV sites/folds
    for site in list_sites:
        print("running site : ", site)
        
        # get training and testing ROI dataframes (contains participant_id + TIV in addition to 330 ROIs)
        df_tr_ = VBMdf[VBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][0])]
        df_te_ = VBMdf[VBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][1])]

        if random_labels and compute_shap: 
            print("permuting the labels ...")
            y = participants_VBM["dx"].values
            y = np.random.permutation(y)
            participants_VBM["dx"]=y
                
        y_train = pd.merge(df_tr_, participants_VBM, on ="participant_id")["dx"].values
        y_test = pd.merge(df_te_, participants_VBM, on ="participant_id")["dx"].values
        
        # find index in participants df of the train and test subjects for the current LOSO CV site and train data size
        train = list(participants_VBM.index[participants_VBM['participant_id'].isin(splits[site+"-"+str(datasize)][0])])
        test = list(participants_VBM.index[participants_VBM['participant_id'].isin(splits[site+"-"+str(datasize)][1])])
        
        assert list(y_train)==list(participants_VBM.iloc[train]["dx"].values)
        assert list(y_test)==list(participants_VBM.iloc[test]["dx"].values)
        participants_te = df_te_["participant_id"].values
        participants_tr = df_tr_["participant_id"].values

        # drop participant_ids , sessions, and global measures 
        # (TIV, total cerebrosplinal fluid, gray matter, and white matter volumes)
        exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
        df_tr_ = df_tr_.drop(columns=exclude_elements)
        df_te_ = df_te_.drop(columns=exclude_elements)

        df_tr_ = remove_zeros(df_tr_)
        df_te_ = remove_zeros(df_te_) 

        X_train = df_tr_.values
        X_test = df_te_.values

        assert list(df_tr_.columns) == list(df_te_.columns)
        
        # df_te_= df_te_[["rPal_GM_Vol","lPal_GM_Vol"]] #[["lPut_GM_Vol","rPut_GM_Vol",
        # df_tr_= df_tr_[["rPal_GM_Vol","lPal_GM_Vol"]]

        # get classifier
        classifier = get_classifier(classif)

        # fit residualizer
        residualizer.fit(X_train, Zres[train])
        X_train = residualizer.transform(X_train, Zres[train])
        X_test = residualizer.transform(X_test, Zres[test])

        # fit scaler
        scaler_ = StandardScaler()
        X_train = scaler_.fit_transform(X_train)
        X_test = scaler_.transform(X_test)

        classifier.fit(X_train, y_train)
        # classifier.fit(tr_,np.random.permutation(y_train))
        y_pred = classifier.predict(X_test)

        # get classification scores for current classifier
        score_test = get_scores(classifier, X_test)
        score_train = get_scores(classifier, X_train)

        if compute_shap :
            assert datasize==800, "shap values should be computed only at maximum training set size!"
            assert classif=="svm", "shap values should be computed with the best performing classifier, the SVM-RBF"
            
            print("SHAP step : predict proba")
            # here, we can compute shap values different ways:
            # ideally, we take the whole training dataset as background data to estimate the shap values
            # for a non-linear model like the SVM-RBF, which means setting background_data = X_train 
            # otherwise, for computational efficiency (using the whole training data is time consuming),
            # we can use a subset of the training data, either using kmeans clustering 
            # (weighted by the nb of subjects in each cluster), which we can do with 
            # background_data = shap.kmeans(X_train, 100), or we can subsample randomly from Xtrain without replacement 
            # with: X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
            
            background_data = X_train 
            print("background data shape (X train)", np.shape(background_data))

            explainer = shap.KernelExplainer(classifier.best_estimator_.decision_function, background_data)
            shap_values = joblib.Parallel(n_jobs=5)(
                joblib.delayed(explainer)(x) for x in tqdm(X_test)
            )

            # if you do not wish to parallelize the computations (not necessary if the background_data is not the
            # whole training set), do : shap_values = explainer.shap_values(X_test)
            create_folder_if_not_exists(SHAP_DIR)
            shapfile = SHAP_DIR+"ShapValues_VBM_SVM_RBF_"+site+"_background_alltr_parallelized"+randomizedlabels_str+"_avril25"

            save_shap_file(shap_values, shapfile)

            # shap.summary_plot(shap_values, X_test, feature_names=[f'Feature {i}' for i in range(X_test.shape[1])])
            print("shape and type shap values : ",type(shap_values), np.shape(shap_values))

        
        dict_score_by_site[site] = {"score test": score_test,"score train": score_train, \
                                    "participant_ids_te":participants_te, "participant_ids_tr":participants_tr}

        roc_auc = roc_auc_score(y_test, score_test)
        bacc = balanced_accuracy_score(y_test, y_pred)

        roc_auc_list.append(roc_auc)
        scores_list.append(score_test)
        bacc_list.append(bacc)
        print("...done.")

        metrics_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc}
        print(site, " roc auc ", round(100*roc_auc,2), " bacc ", round(100*bacc,2))
    
    if onesite is not None :
        metrics_dict["mean over all sites"] = {"roc_auc": np.mean(roc_auc_list), "balanced-accuracy":np.mean(bacc_list)}
        print("MEAN : roc_auc ", np.round(100*np.mean(roc_auc_list),2), "balanced-accuracy ",np.round(100*np.mean(bacc_list),2))
        print("STD : roc_auc ", np.round(100*np.std(roc_auc_list),2), "balanced-accuracy ",np.round(100*np.std(bacc_list),2))

    if save and classif=="svm" and not N763 and not random_labels and onesite is None:
        create_folder_if_not_exists(RESULTSFOLDER+"/stacking")
        create_folder_if_not_exists(RESULTSFOLDER+"/stacking/SVMRBF_VBMROI")
        scores_filepath = RESULTSFOLDER+"stacking/SVMRBF_VBMROI/scores_tr_te_N861_train_size_N"+str(datasize)+".pkl"
        print(scores_filepath)
        if not os.path.exists(scores_filepath):
            print("saving scores for stacking ...")
            save_pkl(dict_score_by_site, scores_filepath)
    
    # if save and not random_labels and onesite is None:
    #     quit()
    #     if N763: strN = '_N763'
    #     else: strN = '_N861'
    #     create_folder_if_not_exists(RESULTSFOLDER+"classifVBM/")
    #     results_file = RESULTSFOLDER+"classifVBM/"+classif+"_N"+str(datasize)+"_"+atlas+"_VBM_ROI"+strN+".pkl"
    #     print("\nsaving classification results ...")
    #     save_pkl(metrics_dict, results_file)

    return dict_score_by_site

def print_info_participants():
    splits = get_LOSO_CV_splits_N861()    
     # read participants dataframe
    participants = get_participants()
     # prepare residualizer for residualization on age, sex, and site 
    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    # select the participants for VBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    participants_all = list(splits["Baltimore-"+str(800)][0])+list(splits["Baltimore-"+str(800)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_VBM = participants.iloc[msk]   
    participants_VBM = participants_VBM.reset_index(drop=True)

    value_counts = participants_VBM['dx'].value_counts()
    total_values = len(participants_VBM['dx'])
    percentages = (value_counts / total_values) * 100
    print(f"Total percentage of BD: {percentages[1]:.2f}%")
    print(f"Total percentage of HC: {percentages[0]:.2f}%")

    value_counts_sex = participants_VBM['sex'].value_counts()
    total_values_sex = len(participants_VBM['sex'])
    percentages_sex = (value_counts_sex / total_values_sex) * 100
    print(f"Total percentage of 1: {percentages_sex[1]:.2f}%")
    print(f"Total percentage of 0: {percentages_sex[0]:.2f}%")

    print("mean age and std ", participants_VBM["age"].mean(), "  ", participants_VBM["age"].std())
    print("number of participants ", len(participants_VBM["participant_id"].unique()))

    print("\nnb participants per site :")
    for site in get_predict_sites_list():
        print(site , " nb of participants :",len(participants_VBM[participants_VBM["site"]==site]["participant_id"]))


def main():
    
    # to compute shap values at maximum training set size with the dataset containing the most subjects (N=861)
    # for the best-performing classifier using VBM ROI features (SVM-RBF)
    # onesite_ to choose from ["Baltimore", "Boston", "Dallas", "Detroit", "Hartford",
    #  "mannheim", "creteil", "udine", "galway", "pittsburgh", "grenoble", "geneve"]
    onesite_ = "grenoble"
    classif_vbm_ROI(classif = "svm", datasize = 800, save = False, N763=False, compute_shap=True, random_labels=False, onesite=onesite_) 
    quit()
    for i in range(5):
        start_time = time.time()
        print(onesite_)
        classif_vbm_ROI(classif = "svm", datasize = 800, save = False, N763=False, compute_shap=True, random_labels=True, onesite=onesite_) 
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print("site : ",onesite_)
        print(f"The function took {hours}h {minutes}m {seconds}s to run.") 
    quit()
    classif_vbm_ROI(classif = "svm", datasize = 800, save = True, N763=False) 
    quit()
    # to compute the classificaitons for all ML models and all training set sizes :

    # for the dataset with all VBM ROI measures and all subjects (N=861)
    for size in [100,175,250,350,400,500,600,700,800] :
        for classifier in ["svm"]:#, "MLP", "EN", "L2LR", "xgboost"]: 
            classif_vbm_ROI(classif = classifier, datasize = size, save = True, N763=False)
    quit()

    # for the dataset with subjects that have both measures for SBM ROI and VBM ROI (N=763)
    for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        for classifier in ["svm", "MLP", "EN", "L2LR", "xgboost"]: 
            classif_vbm_ROI(classif = classifier, datasize = size, save = False, N763=True)


if __name__ == '__main__':
    main()

"""
Nmax=861 for SVM-RBF 
MEAN : roc_auc  72.58 balanced-accuracy  65.98
STD : roc_auc  8.03 balanced-accuracy  6.72

Nmax=763 for SVM-RBF 
MEAN : roc_auc  73.41 balanced-accuracy  68.71
STD : roc_auc  8.06 balanced-accuracy  7.14

"""
