import os , pandas as pd, numpy as np, json, gc, sys
import pickle, shap, joblib, time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import pandas as pd
from utils import standard_error, save_pkl, read_pkl, \
    get_participants, get_predict_sites_list, has_nan, has_zeros_col, get_classifier,\
        get_LOSO_CV_splits_N763, get_scores, create_folder_if_not_exists, get_LOSO_CV_splits_N861, save_shap_file

sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

#inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATAFOLDER=ROOT+"data/processed/"
#outputs
RESULTSFOLDER=ROOT+"results_classif/classifSBM/"
SHAP_DIR=ROOT+"models/ShapValues/shap_computed_from_all_Xtrain/"

def get_N861(SBMdf):
    splits = get_LOSO_CV_splits_N861()    
    # read participants dataframe
    participants = get_participants()
    # select the participants for VBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    participants_all = list(splits["Baltimore-"+str(800)][0])+list(splits["Baltimore-"+str(800)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_VBM = participants.iloc[msk]   
    participants_VBM = participants_VBM.reset_index(drop=True)

    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    residualizer = Residualizer(data=participants_VBM, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants_VBM)

    VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    VBMdf["participant_id"] = VBMdf["participant_id"].str.removeprefix("sub-") 
    # reorder VBMdf to have rows in the same order as participants_VBM
    VBMdf = VBMdf.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()

    SBMdf = pd.merge(SBMdf, participants_VBM[["participant_id","dx"]], on ="participant_id")
    SBMroi = [roi for roi in list(SBMdf.columns) if roi!="participant_id" and roi!="dx"]
    assert len(SBMroi)==331 # 296 ROI + 34 subcortical ROI + TIV

    # get mean of SBM values across all HC and BD subjects
    HC_means = SBMdf[SBMdf["dx"]==0][SBMroi].mean()
    BD_means = SBMdf[SBMdf["dx"]==1][SBMroi].mean()
    # convert to a one-row dataframe
    HC_means = HC_means.to_frame().T  
    BD_means = BD_means.to_frame().T  

    #participants with VBM values and not SBM values (861-763=98 participants)
    participantsVBM_no_SBM = [p for p in VBMdf["participant_id"].values if p not in SBMdf["participant_id"].values]
    participantsVBM_no_SBM_HC = [p for p in participantsVBM_no_SBM if p in list(participants_VBM[participants_VBM["dx"]==0]["participant_id"].values)]
    participantsVBM_no_SBM_BD = [p for p in participantsVBM_no_SBM if p in list(participants_VBM[participants_VBM["dx"]==1]["participant_id"].values)]
    assert len(participantsVBM_no_SBM_HC)+len(participantsVBM_no_SBM_BD) == len(participantsVBM_no_SBM)

    # Repeat the single row len(participantsVBM_no_SBM_HC) times and add "participant_id"
    HC_means = HC_means.loc[HC_means.index.repeat(len(participantsVBM_no_SBM_HC))].reset_index(drop=True)
    HC_means.insert(0, "participant_id", participantsVBM_no_SBM_HC) 
    HC_means["dx"] = 0
 
    BD_means = BD_means.loc[BD_means.index.repeat(len(participantsVBM_no_SBM_BD))].reset_index(drop=True)
    BD_means.insert(0, "participant_id", participantsVBM_no_SBM_BD)  
    BD_means["dx"] = 1

    newSBM = pd.concat([HC_means,BD_means],axis=0)
    SBMdf_861 = pd.concat([SBMdf, newSBM],axis=0)
    assert len(SBMdf_861)==861
    SBMdf_861 = SBMdf_861.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()
    print(SBMdf_861)
    print(participants_VBM)
    quit()

    return SBMdf_861, participants_VBM, splits

def classif_sbm_ROI(classif, datasize, save=False, compute_shap=False, atlas="Destrieux", include_subcorticalROI=True,\
        seven_subcortical_Nunes_replicate=False, classif_augmented_SBMROI=False, random_labels=False, onesite=None):
    """
        classif : (str) classifier name (it has to be "MLP","L2LR","svm","xgboost", or "EN")
        datasize : (int) has to be 75, 150, 200, 300, 400, 450, 500, 600, or 700
        save : (bool) whether we save the pkl file containing roc auc and balanced accuracy measures for each LOSO-CV site
        compute_shap: (bool) wether we compute shap values. only applied for maximum training set size and the best-performing classifier
                for SBM ROI, which is an elastic net ("EN" for "classif" variable)
        atlas : (str) "Destrieux" or "Desikan", the atlas name used to generate SBM ROI 
        include_subcortical : (bool), whether we inlcude all subcortical measures to the SBM ROI for classification 
                the alternative is having only cortical thickness and surface area measures. 
        random_labels : (bool) use random labels for training and testing to compute shap with random labels
        onesite : (None or str) to compute Shap one site at a time (onesite is the name of said site) to compute shap 
                for each site in parallel if needed
        classif_augmented_SBMROI : (bool) use SBM ROI with data imputation (only for stacking for the meta-model)
    """

    assert atlas in ["Desikan", "Destrieux"], "wrong atlas name!"
    assert classif in ["MLP","L2LR","svm","xgboost","EN"], "wrong classifier name"
    assert not (seven_subcortical_Nunes_replicate and not include_subcorticalROI)
    if classif_augmented_SBMROI: assert classif=="EN" and atlas=="Destrieux" and include_subcorticalROI and \
        not seven_subcortical_Nunes_replicate, "not the right parameters for SBM ROI classification for stacking"
    if classif_augmented_SBMROI: assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"
    else: assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"

    #verifications for shap computations
    if onesite is not None: assert onesite in get_predict_sites_list()
    if random_labels: randomizedlabels_str="_randomized_labels"
    else: randomizedlabels_str=""

    # read splits
    splits = get_LOSO_CV_splits_N763()

    roc_auc_list, scores_list, bacc_list = [],[],[]
    metrics_dict, coefs_dict = {} , {}

    # read participants dataframe
    participants = get_participants()

    # prepare residualizer for residualization on age, sex, and site 
    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    # select the participants for SBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    participants_all = list(splits["Baltimore-"+str(700)][0])+list(splits["Baltimore-"+str(700)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_SBM = participants.iloc[msk]   
    participants_SBM = participants_SBM.reset_index(drop=True)

    residualizer = Residualizer(data=participants_SBM, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants_SBM)

    dict_score_by_site = dict()
    Xim_test_all_sites = []
    
    # seven_subcortical_Nunes_replicate changes the file name to SBMROI_<atlas>_CT_SA_subcortical_N763.csv to get 34 subcortical ROIs instead of the 
    # same 7 subcortical ROIs as in Nunes et al.
    if seven_subcortical_Nunes_replicate: str_7ROI = "_7ROIsub"
    else : str_7ROI = ""
    SBMdf = pd.read_csv(DATAFOLDER+"SBMROI_"+atlas+"_CT_SA_subcortical"+str_7ROI+"_N763.csv")

    # reorder SBMdf to have rows in the same order as participants_SBM
    SBMdf = SBMdf.set_index('participant_id').reindex(participants_SBM["participant_id"].values).reset_index()

    if classif_augmented_SBMROI: SBMdf, participants_SBM, splits = get_N861(SBMdf)

    if onesite is not None: list_sites = [onesite]
    else : list_sites = get_predict_sites_list()
   
    for site in list_sites: 
        print("running site : ",site)
    
        # get training and testing ROI dataframes (contains participant_id + TIV in addition to 330 ROIs)
        df_tr_ = SBMdf[SBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][0])]
        df_te_ = SBMdf[SBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][1])]

        if random_labels and compute_shap: 
            print("permuting the labels ...")
            y = participants_SBM["dx"].values
            y = np.random.permutation(y)
            participants_SBM["dx"]=y
                
        y_train = pd.merge(df_tr_, participants_SBM, on ="participant_id")["dx"].values
        y_test = pd.merge(df_te_, participants_SBM, on ="participant_id")["dx"].values
        
        # find index in participants df of the train and test subjects for the current LOSO CV site and train data size
        train = list(participants_SBM.index[participants_SBM['participant_id'].isin(splits[site+"-"+str(datasize)][0])])
        test = list(participants_SBM.index[participants_SBM['participant_id'].isin(splits[site+"-"+str(datasize)][1])])
        
        assert list(y_train)==list(participants_SBM.iloc[train]["dx"].values)
        assert list(y_test)==list(participants_SBM.iloc[test]["dx"].values)
        participants_te = df_te_["participant_id"].values
        participants_tr = df_tr_["participant_id"].values

        # drop participant_ids and TIV measures
        df_tr_ = df_tr_.drop(columns=["participant_id", "TIV"])
        df_te_ = df_te_.drop(columns=["participant_id", "TIV"])

        X_train = df_tr_.values
        X_test = df_te_.values

        # get ROI names in a list
        assert list(df_tr_.columns) == list(df_te_.columns)
        roi_names = list(df_te_.columns)
        
        if not include_subcorticalROI: 
            roi_names = [roi for roi in roi_names if roi.endswith("_thickness") or roi.endswith("_area")]
            if atlas=="Destrieux": assert len(roi_names) == 296 # 74 for each ROI type (area and cortical thickness) for both hemispheres
            if atlas=="Desikan": assert len(roi_names) == 136 # 34 for each ROI type (area and cortical thickness) for both hemispheres
            X_train = df_tr_[roi_names].values
            X_test = df_te_[roi_names].values
        
        # print("shape train: ",np.shape(X_train), ", test: ", np.shape(X_test))
        # print("type train: ",type(X_train), ", test: ", type(X_test))
        # print("roi_names : ", np.shape(roi_names), type(roi_names), " len ", len(roi_names))

        assert not has_nan(X_train) and not has_nan(X_test)
        if has_zeros_col(X_test) : 
            print("X_test has columns of only zeros")
            zero_columns_te = df_te_.columns[(df_te_ == 0).all()]
            # print(df_te_[zero_columns_te])
            if zero_columns_te.tolist() != ['5th-Ventricle']: 
                print(df_te_[zero_columns_te])
                quit()

        if has_zeros_col(X_train) : 
            print("X_train has columns of only zeros")
            zero_columns_tr = df_tr_.columns[(df_tr_ == 0).all()]
            if zero_columns_tr.tolist() != ['5th-Ventricle']: 
                print(df_te_[zero_columns_tr])
                quit()

        # get classifier
        classifier = get_classifier(classif)

        # fit residualizer
        residualizer.fit(X_train, Zres[train])
        X_train = residualizer.transform(X_train, Zres[train])
        X_test = residualizer.transform(X_test, Zres[test])

        # scale
        scaler_ = StandardScaler()
        X_train = scaler_.fit_transform(X_train)
        X_test = scaler_.transform(X_test)

        # fit
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if classif=="EN" and datasize==700: 
            best_model = classifier.best_estimator_
            coefficients = best_model.coef_
            print(np.shape(coefficients), type(coefficients))
            coefs_dict[site]=coefficients

        
        # get classification scores for current classifier
        score_test = get_scores(classifier, X_test)
        score_train = get_scores(classifier, X_train)

        if compute_shap :
            assert datasize==700, "shap values should be computed only at maximum training set size!"
            assert classif=="EN", "shap values should be computed with the elastic net as it is the classifier that performs best for Freesurfer ROI data"
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
            explainer = shap.Explainer(classifier.best_estimator_, background_data)
            shap_values = explainer.shap_values(X_test)

            # if you do not wish to parallelize the computations (not necessary if the background_data is not the
            # whole training set), do : shap_values = explainer.shap_values(X_test)
            create_folder_if_not_exists(SHAP_DIR)
            shapfile = SHAP_DIR+"ShapValues_VBM_SVM_RBF_"+site+"_background_alltr_parallelized"+randomizedlabels_str+"_"+atlas

            save_shap_file(shap_values, shapfile)
            print("shape and type shap values : ",type(shap_values), np.shape(shap_values))

                
        roc_auc = roc_auc_score(y_test, score_test)
        bacc = balanced_accuracy_score(y_test, y_pred)

        roc_auc_list.append(roc_auc)
        scores_list.append(score_test)
        Xim_test_all_sites.append(X_test)
        bacc_list.append(bacc)
        print("...done.")
        metrics_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc}
        dict_score_by_site[site] = {"score test": score_test,"score train": score_train, \
                                    "participant_ids_te":participants_te, "participant_ids_tr":participants_tr}
        print(site, " roc auc ", round(100*roc_auc,2), " bacc ", round(100*bacc,2))
    
    if onesite is not None :
        metrics_dict["mean over all sites"] = {"roc_auc": np.mean(roc_auc_list), "balanced-accuracy":np.mean(bacc_list)}
        print("MEAN : roc_auc ", np.round(100*np.mean(roc_auc_list),2), "balanced-accuracy ",np.round(100*np.mean(bacc_list),2))
        print("STD : roc_auc ", np.round(100*np.std(roc_auc_list),2), "balanced-accuracy ",np.round(100*np.std(bacc_list),2))

    print("\n",classif , "roc auc mean % :", round(100*np.mean(roc_auc_list,axis=0),2))
    print("bacc mean % :", round(100*np.mean(bacc_list,axis=0),2))
    se_roc_auc = standard_error(roc_auc_list)
    se_bacc = standard_error(bacc_list)

    print("ROC-AUC Standard Error:", np.round(se_roc_auc,3))
    print("ROC-AUC Standard Deviation:", np.round(100*np.std(roc_auc_list),2))

    print("Balanced Accuracy Standard Error:", np.round(se_bacc,3))
    metrics_dict["mean over all sites"] = {"roc_auc": np.mean(roc_auc_list), "balanced-accuracy":np.mean(bacc_list)}

    # if classif=="EN" and datasize ==700: save_pkl(coefs_dict, "coefficientsdictEN700_"+atlas+sub_roi+"_janv25.pkl")
    if not include_subcorticalROI: 
        strsub = "_no_subcortical"
        folder = "no_subcortical/"
    else: 
        if seven_subcortical_Nunes_replicate: 
            strsub = "_7subROI"
            folder = "with7subcorticalROI/"
        else : 
            strsub = ""
            folder = ""
    results_file = RESULTSFOLDER+folder+str(classif)+"_N"+str(datasize)+"_"+atlas+"_SBM_ROI"+strsub+"_N763.pkl"

    if classif=="EN" and atlas=="Destrieux" and include_subcorticalROI and \
        not seven_subcortical_Nunes_replicate :
        create_folder_if_not_exists(RESULTSFOLDER+"/stacking")
        create_folder_if_not_exists(RESULTSFOLDER+"/stacking/EN_SBMROI")
        scores_filepath = RESULTSFOLDER+"/stacking/EN_SBMROI/scores_tr_te_N763_train_size_N"+str(datasize)+".pkl"
        if not os.path.exists(scores_filepath):
            print("saving scores for stacking ...")
            save_pkl(dict_score_by_site, scores_filepath)
    
    if save:
        print("\nsaving classification results ...")
        save_pkl(metrics_dict, results_file)

    return dict_score_by_site

def print_info_participants():
    """
        Aim: print information on SBM ROI dataset 

    """

    # read splits
    splits = get_LOSO_CV_splits_N763()

     # read participants dataframe
    participants = get_participants()
    participants_all = list(splits["Baltimore-"+str(700)][0])+list(splits["Baltimore-"+str(700)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_SBM = participants.iloc[msk]   
    participants_SBM = participants_SBM.reset_index(drop=True)

    print(participants_SBM)

    value_counts = participants_SBM['dx'].value_counts()
    total_values = len(participants_SBM['dx'])
    percentages = (value_counts / total_values) * 100
    print(f"Total percentage of BD: {percentages[1]:.2f}%", " counts : ",value_counts[1])
    print(f"Total percentage of HC: {percentages[0]:.2f}%", " counts : ",value_counts[0])

    value_counts_sex = participants_SBM['sex'].value_counts()
    total_values_sex = len(participants_SBM['sex'])
    percentages_sex = (value_counts_sex / total_values_sex) * 100
    print(f"Percentage of females: {percentages_sex[1]:.2f}%" ," counts : ", value_counts_sex[1])
    print(f"Percentage of males: {percentages_sex[0]:.2f}%", " counts : ", value_counts_sex[0])

    print("mean age and std ", participants_SBM["age"].mean(), "  ", participants_SBM["age"].std())
    print("number of participants ", len(participants_SBM["participant_id"].unique()))

    print("\nnb participants per site :")
    for site in get_predict_sites_list():
        df_by_site = participants_SBM[participants_SBM["site"]==site]
        nb_participants_by_site = len(df_by_site["dx"])
        print(site , " nb of participants :",nb_participants_by_site)
        print(site , " mean age of participants :",round(np.mean(df_by_site["age"])))
        print(site , " std age of participants :", round(np.std(df_by_site["age"])))
        print(site , " nb of female participants :",len(df_by_site[df_by_site["sex"]==1.0]))
        print(site , " percentage of female participants :",round(100*(len(df_by_site[df_by_site["sex"]==1.0])/nb_participants_by_site)))
        print(site , " nb of BD participants :",len(df_by_site[df_by_site["dx"]==1]),"\n")
        print(site , " percentage of BD participants :",round(100*(len(df_by_site[df_by_site["dx"]==1])/nb_participants_by_site)),"\n")


def read_results_classif(datasize=700, atlas="Destrieux"):
    """
        datasize : (int) size of training data 
        atlas : (str) atlas name "Destrieux" or "Desikan
    """
    assert atlas in ["Desikan", "Destrieux"], "wrong atlas name!"
    roc_auc = {}
    for classif in ["L2LR", "EN","svm","xgboost",  "MLP"]:
        mean_auc = []
        for site in get_predict_sites_list():
            data = read_pkl(RESULTSFOLDER+classif+"_N"+str(datasize)+"_"+atlas+"_SBM_ROI.pkl")
            mean_auc.append(data[site]["roc_auc"])
        roc_auc[classif]=np.mean(mean_auc)
    print(roc_auc)


def main():
    
    # to compute shap values at maximum training set size with the dataset containing the most subjects (N=861)
    # for the best-performing classifier using VBM ROI features (SVM-RBF)
    # onesite_ to choose from ["Baltimore", "Boston", "Dallas", "Detroit", "Hartford",
    #  "mannheim", "creteil", "udine", "galway", "pittsburgh", "grenoble", "geneve"]
    start_time = time.time()
    for onesite_ in get_predict_sites_list():
        for i in range(30):
            
            print(onesite_)
            classif_sbm_ROI(classif = "EN", datasize = 700, save = False, compute_shap=True, random_labels=False, onesite=onesite_) 
            classif_sbm_ROI(classif = "EN", datasize = 700, save = False, compute_shap=True, random_labels=True, onesite=onesite_) 
            
            print("site : ",onesite_)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"The function took {hours}h {minutes}m {seconds}s to run.") 
    quit()
    """
    # to print demographic information on BIOBD/BSNIP subjects with SBM ROI (N=763)
    print_info_participants()

    # to run the classification with the best-performing classifier (EN) and maximum training set size : 
    classif_sbm_ROI("EN",700, atlas="Destrieux", include_subcorticalROI=True, compute_shap=True, save=False)
    """
    classif_sbm_ROI("EN", 800, atlas="Destrieux", include_subcorticalROI=True, classif_augmented_SBMROI=True)

    for size in [100,175,250,350,400,500,600,700,800]:
        classif_sbm_ROI("EN", size, atlas="Destrieux", include_subcorticalROI=True, classif_augmented_SBMROI=True)
    quit()
    # to run classification for all training set sizes and all 5 ML classifiers for Destrieux atlas with subcortical ROI
    for atlas in ["Destrieux"]:
        for include_subcorticalROI in [True]:
            for classif_ in ["L2LR", "EN","svm","xgboost", "MLP"]: 
                for trainingdatasize in  [75, 150, 200, 300, 400, 450, 500, 600, 700]:
                    classif_sbm_ROI(classif_, trainingdatasize, atlas=atlas, include_subcorticalROI=include_subcorticalROI, seven_subcortical_Nunes_replicate=True, save=True)
    """
    # to print the results of classification for different training set sizes
    for trainingdatasize in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        print("training datasize ", trainingdatasize)
        read_results_classif(trainingdatasize)
    """
    
if __name__ == "__main__":
    main()

