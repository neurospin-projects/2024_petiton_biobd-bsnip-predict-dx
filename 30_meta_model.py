import numpy as np, pandas as pd
from utils import read_pkl, get_predict_sites_list, get_scores_pipeline, save_pkl

# for linear model (= scikit learn imports)
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from scipy.stats import ttest_rel

# inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
LEARNING_CRV_TL_DIR = ROOT+"models/TL/TL_learningcurve/"
PATHS_VBM_763_VOXELWISE=[ROOT+f"N763_classif_VBM_TL_run{i}" for i in range(1,6)]

# ouputs
RESULTS_DICT = ROOT+"reports/results_classif/meta_model/summary_dict_results.pkl"
RESULTS_DICT_N763 = ROOT+"reports/results_classif/meta_model/summary_dict_results_N763_no_imputation.pkl"

def create_index_mapping(ids_861, ids_763):
    """
    Create mapping to transform array ordered by participant_id list of N861 to array of participant_ids of N763
    """
    # Create mapping from ID to index in both lists
    ids_861=ids_861.tolist()
    ids_763=ids_763.tolist()

    id_to_index = {id_val: idx for idx, id_val in enumerate(ids_861)}
    
    # Create the transformation mapping
    # For each position in the N763 array, find where that ID is in the N861 array
    indices_861to763 = []
    for id in ids_763:
        index861 = id_to_index[id]
        indices_861to763.append(index861)
    
    return indices_861to763

def classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr=None, ypred_voxelwise_vbm_te=None,\
    ypred_voxelwise_sbm_tr = None, ypred_voxelwise_sbm_te = None, combination_nb=None, N763=False, onlydeep=False):

    # check we have the same participant_id lists for train and test sets
    if N763: # reordering necessary in this case
        indices_tr = create_index_mapping(sbm_current_site["participant_ids_tr"], vbm_current_site["participant_ids_tr"])
        indices_te = create_index_mapping(sbm_current_site["participant_ids_te"], vbm_current_site["participant_ids_te"])
        sbm_current_site["participant_ids_te"] = sbm_current_site["participant_ids_te"][indices_te]
        sbm_current_site["participant_ids_tr"] = sbm_current_site["participant_ids_tr"][indices_tr]
        sbm_current_site["score train"] = sbm_current_site["score train"][indices_tr]
        sbm_current_site["score test"] = sbm_current_site["score test"][indices_te]
        sbm_current_site["y_train"] = sbm_current_site["y_train"][indices_tr]
        sbm_current_site["y_test"] = sbm_current_site["y_test"][indices_te]
        if ypred_voxelwise_sbm_tr is not None: 
            ypred_voxelwise_sbm_tr = ypred_voxelwise_sbm_tr[indices_tr]
            ypred_voxelwise_sbm_te = ypred_voxelwise_sbm_te[indices_te]


    assert np.all (sbm_current_site["participant_ids_tr"] == vbm_current_site["participant_ids_tr"] )
    assert np.all (sbm_current_site["participant_ids_te"] == vbm_current_site["participant_ids_te"])

    sbm_score_tr = sbm_current_site["score train"]
    vbm_score_tr = vbm_current_site["score train"]
    assert len(sbm_score_tr)==len(vbm_score_tr) # more checks just in case

    sbm_score_te = sbm_current_site["score test"]
    vbm_score_te = vbm_current_site["score test"]
    assert len(sbm_score_te)==len(vbm_score_te)

    sbm_y_tr = sbm_current_site["y_train"]
    sbm_y_te = sbm_current_site["y_test"]

    assert np.all(sbm_y_tr == vbm_current_site["y_train"])
    assert np.all(sbm_y_te == vbm_current_site["y_test"])
    
    if onlydeep: X_train = np.column_stack((ypred_voxelwise_sbm_tr, ypred_voxelwise_vbm_tr))
    else: X_train = np.column_stack((sbm_score_tr, vbm_score_tr)) 

    if ypred_voxelwise_vbm_tr is not None and not onlydeep:  X_train = np.column_stack((X_train, ypred_voxelwise_vbm_tr)) 
    if ypred_voxelwise_sbm_tr is not None and not onlydeep: X_train = np.column_stack((X_train, ypred_voxelwise_sbm_tr)) 

    y_train = sbm_y_tr

    # # print("X_train and ytrain shapes : ", X_train.shape, "  ",y_train.shape) 
    if onlydeep: X_test = np.column_stack((ypred_voxelwise_sbm_te, ypred_voxelwise_vbm_te))
    else : X_test = np.column_stack((sbm_score_te, vbm_score_te))  
    if ypred_voxelwise_vbm_te is not None and not onlydeep:  X_test = np.column_stack((X_test, ypred_voxelwise_vbm_te)) 
    if ypred_voxelwise_sbm_te is not None and not onlydeep:  X_test = np.column_stack((X_test, ypred_voxelwise_sbm_te)) 

    y_test = sbm_y_te
    # print("X_test and ytest shapes : ",X_test.shape, "  ", y_test.shape) 

    # param_grid = {'C': [0.01, 0.1, 1., 10.], 'penalty': ['l2'],'class_weight':['balanced']}
    # classifier = GridSearchCV(estimator=lm.LogisticRegression(fit_intercept=False), param_grid=param_grid, cv=3)
    classifier = lm.LogisticRegression(C=0.01, class_weight='balanced', fit_intercept=False)

    # Pipeline
    pipe = make_pipeline(
        StandardScaler(),
        classifier
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # best_params = pipe.best_params_
    # print("Best Parameters:", best_params)

    # get classification scores for current classifier
    score_test = get_scores_pipeline(pipe, X_test)
    score_train = get_scores_pipeline(pipe, X_train)

    roc_auc = roc_auc_score(y_test, score_test)
    bacc = balanced_accuracy_score(y_test, y_pred)

    # print("roc auc for site ",site, ": ",roc_auc)

    results_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc,"score test": score_test,"score train": score_train, \
                                "participant_ids_te":sbm_current_site["participant_ids_te"], "participant_ids_tr":sbm_current_site["participant_ids_tr"],\
                                    "y_train": y_train, "y_test":y_test}

    if ypred_voxelwise_vbm_tr is not None and ypred_voxelwise_vbm_te is not None and combination_nb:
        results_dict[site]["combination_nb"]=combination_nb

def classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr=None, ypred_voxelwise_vbm_te=None,\
    ypred_voxelwise_sbm_tr = None, ypred_voxelwise_sbm_te = None, combination_nb=None, onlydeep=False):


    # assert np.all (sbm_current_site["participant_ids_tr"] == vbm_current_site["participant_ids_tr"] )
    # assert np.all (sbm_current_site["participant_ids_te"] == vbm_current_site["participant_ids_te"])

    sbm_score_tr = sbm_current_site["score train"]
    vbm_score_tr = vbm_current_site["score train"]
    assert len(sbm_score_tr)==len(vbm_score_tr) # more checks just in case

    sbm_score_te = sbm_current_site["score test"]
    vbm_score_te = vbm_current_site["score test"]
    assert len(sbm_score_te)==len(vbm_score_te)

    sbm_y_tr = sbm_current_site["y_train"]
    sbm_y_te = sbm_current_site["y_test"]

    # assert np.all(sbm_y_tr == vbm_current_site["y_train"])
    # assert np.all(sbm_y_te == vbm_current_site["y_test"])
    
    if onlydeep: X_train = np.column_stack((ypred_voxelwise_sbm_tr, ypred_voxelwise_vbm_tr))
    else: X_train = np.column_stack((sbm_score_tr, vbm_score_tr)) 

    if ypred_voxelwise_vbm_tr is not None and not onlydeep:  X_train = np.column_stack((X_train, ypred_voxelwise_vbm_tr)) 
    if ypred_voxelwise_sbm_tr is not None and not onlydeep: X_train = np.column_stack((X_train, ypred_voxelwise_sbm_tr)) 

    y_train = sbm_y_tr

    # # print("X_train and ytrain shapes : ", X_train.shape, "  ",y_train.shape) 
    if onlydeep: X_test = np.column_stack((ypred_voxelwise_sbm_te, ypred_voxelwise_vbm_te))
    else : X_test = np.column_stack((sbm_score_te, vbm_score_te))  
    if ypred_voxelwise_vbm_te is not None and not onlydeep:  X_test = np.column_stack((X_test, ypred_voxelwise_vbm_te)) 
    if ypred_voxelwise_sbm_te is not None and not onlydeep:  X_test = np.column_stack((X_test, ypred_voxelwise_sbm_te)) 

    y_test = sbm_y_te
    # print("X_test and ytest shapes : ",X_test.shape, "  ", y_test.shape) 

    # param_grid = {'C': [0.01, 0.1, 1., 10.], 'penalty': ['l2'],'class_weight':['balanced']}
    # classifier = GridSearchCV(estimator=lm.LogisticRegression(fit_intercept=False), param_grid=param_grid, cv=3)
    classifier = lm.LogisticRegression(C=0.01, class_weight='balanced', fit_intercept=False)

    # Pipeline
    pipe = make_pipeline(
        StandardScaler(),
        classifier
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # best_params = pipe.best_params_
    # print("Best Parameters:", best_params)

    # get classification scores for current classifier
    score_test = get_scores_pipeline(pipe, X_test)
    score_train = get_scores_pipeline(pipe, X_train)

    roc_auc = roc_auc_score(y_test, score_test)
    bacc = balanced_accuracy_score(y_test, y_pred)

    # print("roc auc for site ",site, ": ",roc_auc)

    results_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc,"score test": score_test,"score train": score_train, \
                                "participant_ids_te":sbm_current_site["participant_ids_te"], "participant_ids_tr":sbm_current_site["participant_ids_tr"],\
                                    "y_train": y_train, "y_test":y_test}

    if ypred_voxelwise_vbm_tr is not None and ypred_voxelwise_vbm_te is not None and combination_nb:
        results_dict[site]["combination_nb"]=combination_nb

def get_correspondencies_ids(n=8):
    pathvbmN763 = ROOT+"results_classif/meta_model/scores_tr_te_N763_train_size_N"+str(n)+"_vbmroi.pkl"
    pathvbmN81 = ROOT+"results_classif/meta_model/scores_tr_te_N861_train_size_N"+str(n)+"_vbmroi.pkl"
    vbmN763 = read_pkl(pathvbmN763)
    vbmN861 = read_pkl(pathvbmN81)
    indices = {}
    for site in get_predict_sites_list():
        ids_763_tr = vbmN763[site]["participant_ids_tr"]
        ids_763_te = vbmN763[site]["participant_ids_te"]
        ids_861_tr = vbmN861[site]["participant_ids_tr"]
        ids_861_te = vbmN861[site]["participant_ids_te"]
        indices_tr = create_index_mapping(ids_861_tr, ids_763_tr)
        indices_te = create_index_mapping(ids_861_te, ids_763_te)
        indices[site]={"train":indices_tr, "test":indices_te}
    
    return indices

def run_meta_model(include_voxelwise_vbm=True, include_wholebrain_sbm=False, N763=False, save=False, onlydeep=False):
    """
    runs the meta model using scores of classifiers trained and tested with the ypred_voxelwise_sbm_te features : VBM ROI, VBM voxelwise, SBM ROI
    the corresponding classifiers are an support vector machine with RBF kernel, a 5-DE TL densenet121, and en elastic net, respectively.
    include_voxelwise_vbm (bool) : whether to include voxelwise VBM classifier (5-DE TL model) in the meta model
    N763 (bool) : whether to use classifiers trained on N763 dataset (no data impuation for SBM), or use the whole 861 subjects' data but therefore using 
                    data imputation for SBM ROI (elastic net model), since not all subjects had SBM preprocessing available.
    save (bool) : if True, save the classification results of the meta model
    onlydeep (bool) : use only DL produces scores in the meta model (scores from VBM voxelwise and SBM wholebrain)
    """
    sites= get_predict_sites_list()
    if include_wholebrain_sbm: assert N763, " can't have sbm whole brain with N861 "

    for n in range(9):
        print("n : ", n)
        if N763: pathsbm = ROOT+"results_classif/meta_model/EN_N"+str(n)+"_Destrieux_SBM_ROI_N763.pkl"
        else : pathsbm = ROOT+"results_classif/meta_model/EN_N"+str(n)+"_Destrieux_SBM_ROI_subcortical_N763_data_imputation.pkl"
        sbm = read_pkl(pathsbm)

        if N763: pathvbm = ROOT+"results_classif/meta_model/scores_tr_te_N763_train_size_N"+str(n)+"_vbmroi.pkl"
        else : pathvbm = ROOT+"results_classif/meta_model/scores_tr_te_N861_train_size_N"+str(n)+"_vbmroi.pkl"
        vbm = read_pkl(pathvbm)
        # print(vbm)

        if include_voxelwise_vbm:
            dl_tl_5de_tr = read_pkl(LEARNING_CRV_TL_DIR+"reordered_to_fit_metamodel_mean_ypred_252_combinations_df_TL_densenet_Train_set.pkl")
            dl_tl_5de_te = read_pkl(LEARNING_CRV_TL_DIR+"reordered_to_fit_metamodel_mean_ypred_252_combinations_df_TL_densenet_Test_set.pkl")

        if N763: indices = get_correspondencies_ids(n)

        if include_wholebrain_sbm:
            dl_tl_5de_tr_sbm = read_pkl("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/meta_model_sbm_tr_mean_ypreds_5DETL.pkl")
            dl_tl_5de_te_sbm = read_pkl("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/meta_model_sbm_te_mean_ypreds_5DETL.pkl")
    
        # print(dl_tl_5de_tr)
        results_dict = {}

        if include_voxelwise_vbm or include_wholebrain_sbm:

            mean_auc_over_all_combinations, mean_bacc_over_all_combinations = [], []
            mean_auc_for_each_site = []
            mean_auc_for_each_site_sbm, mean_auc_for_each_site_vbm , mean_auc_for_each_site_voxelwise_vbm= [], [], []

            for combination_nb in range(1,253):
                # print("combination_nb ",combination_nb)
                for site in sites:
                    sbm_current_site = sbm[site]
                    vbm_current_site = vbm[site]
                    if include_voxelwise_vbm :
                        maskvbm_tr = (
                            (dl_tl_5de_tr["site"] == site) &
                            (dl_tl_5de_tr["train_idx"] == n) &
                            (dl_tl_5de_tr["combination"] == combination_nb)
                        )
                        maskvbm_te = (
                            (dl_tl_5de_te["site"] == site) &
                            (dl_tl_5de_te["train_idx"] == n) &
                            (dl_tl_5de_te["combination"] == combination_nb)
                        )
                        dl_tl_5de_tr_one_combination = dl_tl_5de_tr[maskvbm_tr].copy()
                        dl_tl_5de_te_one_combination = dl_tl_5de_te[maskvbm_te].copy()
                        ypred_voxelwise_vbm_tr = dl_tl_5de_tr_one_combination["mean_ypred"].values
                        ypred_voxelwise_vbm_te = dl_tl_5de_te_one_combination["mean_ypred"].values
                        ypred_voxelwise_vbm_tr = ypred_voxelwise_vbm_tr[0]
                        ypred_voxelwise_vbm_te = ypred_voxelwise_vbm_te[0]
                        if N763:  
                            ypred_voxelwise_vbm_tr=ypred_voxelwise_vbm_tr[indices[site]["train"]]
                            ypred_voxelwise_vbm_te=ypred_voxelwise_vbm_te[indices[site]["test"]]

                    if include_wholebrain_sbm:
                        mask_sbm_tr = (
                            (dl_tl_5de_tr_sbm["site"] == site) &
                            (dl_tl_5de_tr_sbm["train_idx"] == n) &
                            (dl_tl_5de_tr_sbm["combination"] == combination_nb)
                        )
                        mask_sbm_te = (
                            (dl_tl_5de_te_sbm["site"] == site) &
                            (dl_tl_5de_te_sbm["train_idx"] == n) &
                            (dl_tl_5de_te_sbm["combination"] == combination_nb)
                        )                    
                        dl_tl_5de_tr_one_combination_sbm = dl_tl_5de_tr_sbm[mask_sbm_tr].copy()
                        dl_tl_5de_te_one_combination_sbm = dl_tl_5de_te_sbm[mask_sbm_te].copy()
                        ypred_voxelwise_sbm_tr = dl_tl_5de_tr_one_combination_sbm["mean_ypred"].values
                        ypred_voxelwise_sbm_te = dl_tl_5de_te_one_combination_sbm["mean_ypred"].values
                        ypred_voxelwise_sbm_tr = ypred_voxelwise_sbm_tr[0][site]
                        ypred_voxelwise_sbm_te = ypred_voxelwise_sbm_te[0][site]
                        # print("ypred SBM te size ",np.shape(ypred_voxelwise_sbm_te),"  ",np.shape(ypred_voxelwise_sbm_tr),ypred_voxelwise_sbm_te)
                        # print("ypred SBM te size ",type(ypred_voxelwise_sbm_te),"  ",type(ypred_voxelwise_sbm_tr))

                    if include_wholebrain_sbm and include_voxelwise_vbm: 
                        classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr, \
                                     ypred_voxelwise_vbm_te, ypred_voxelwise_sbm_tr, ypred_voxelwise_sbm_te, combination_nb=combination_nb,N763=N763, onlydeep=onlydeep)
                    elif include_voxelwise_vbm and not include_wholebrain_sbm : 
                        classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr, \
                                     ypred_voxelwise_vbm_te, combination_nb=combination_nb,N763=N763)
                    elif include_wholebrain_sbm and not include_voxelwise_vbm:
                        classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_sbm_tr=ypred_voxelwise_sbm_tr, \
                                         ypred_voxelwise_sbm_te=ypred_voxelwise_sbm_te, combination_nb=combination_nb,N763=N763)
                        
                        
                    if combination_nb==1:
                        sbm_y_te = sbm_current_site["y_test"]
                        mean_auc_for_each_site_sbm.append(roc_auc_score(sbm_y_te, sbm_current_site["score test"]))
                        mean_auc_for_each_site_vbm.append(roc_auc_score(sbm_y_te, vbm_current_site["score test"]))
                        if include_voxelwise_vbm: 
                            mean_auc_for_each_site_voxelwise_vbm.append(roc_auc_score(sbm_y_te, ypred_voxelwise_vbm_te))
                        
                results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                                    "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites]),\
                                                    "combination_nb":combination_nb}
                
                mean_over_sites_auc = np.mean([results_dict[site]["roc_auc"] for site in sites])
                mean_over_sites_bacc = np.mean([results_dict[site]["balanced-accuracy"] for site in sites])

                mean_auc_for_each_site.append([results_dict[site]["roc_auc"] for site in sites])

                mean_auc_over_all_combinations.append(mean_over_sites_auc)
                mean_bacc_over_all_combinations.append(mean_over_sites_bacc)

                # print("Mean AUC:", mean_over_sites_auc)
                # print("Mean balanced-accuracy:",mean_over_sites_bacc)

            mean_auc_for_each_site = np.array(mean_auc_for_each_site)
            mean_auc_for_each_site = mean_auc_for_each_site.mean(axis=0)

            t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_sbm)
            print("Paired t-test p-value btw sbm roi and meta model :", p_value)

            t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_vbm)
            print("Paired t-test p-value btw vbm roi and meta model :", p_value)

            # t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_voxelwise_vbm)
            # print("Paired t-test p-value btw vbm voxelwise and meta model :", p_value)

            t_stat, p_value = ttest_rel(mean_auc_for_each_site_vbm, mean_auc_for_each_site_sbm)
            print("Paired t-test p-value btw vbm roi and sbm roi :", p_value)

            print("\nMean AUC over all combinations ",np.mean(mean_auc_over_all_combinations))
            print("Mean balanced-accuracy over all combinations ",np.mean(mean_bacc_over_all_combinations))

        else:
            for site in sites:
                sbm_current_site = sbm[site]
                vbm_current_site = vbm[site]
                classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, N763=N763)
            
            results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                                    "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites])}
            print("Mean AUC:", np.mean([results_dict[site]["roc_auc"] for site in sites]))
            print("Mean balanced-accuracy:", np.mean([results_dict[site]["balanced-accuracy"] for site in sites]))

    if save: 
        if N763 : save_pkl(results_dict, RESULTS_DICT_N763)
        else : save_pkl(results_dict,RESULTS_DICT)


def reorder_vbm_roi_to_match_sbm_roi_order(sbm, vbm, sites):
    for site in sites:
        sbm_ids_te = sbm[site]["participant_ids_te"]
        vbm_ids_te = vbm[site]["participant_ids_te"]
        vbm_scores_te = vbm[site]["score test"]
        vbm_yte = vbm[site]["y_test"]

        sbm_ids_tr = sbm[site]["participant_ids_tr"]
        vbm_ids_tr = vbm[site]["participant_ids_tr"]
        vbm_scores_tr = vbm[site]["score train"]
        vbm_ytr = vbm[site]["y_train"]

        # Create mapping from VBM IDs to their indices
        id_to_idx_te = {pid: idx for idx, pid in enumerate(vbm_ids_te)}
        id_to_idx_tr = {pid: idx for idx, pid in enumerate(vbm_ids_tr)}

        # Reorder VBM arrays according to SBM order
        reorder_indices_te = [id_to_idx_te[pid] for pid in sbm_ids_te]
        vbm[site]["participant_ids_te"] = vbm_ids_te[reorder_indices_te]
        vbm[site]["score test"] = vbm_scores_te[reorder_indices_te]
        vbm[site]["y_test"] = vbm_yte[reorder_indices_te]

        reorder_indices_tr = [id_to_idx_tr[pid] for pid in sbm_ids_tr]
        vbm[site]["participant_ids_tr"] = vbm_ids_tr[reorder_indices_tr]
        vbm[site]["score train"] = vbm_scores_tr[reorder_indices_tr]
        vbm[site]["y_train"] = vbm_ytr[reorder_indices_tr]

        return vbm

def run_meta_model_new(include_voxelwise_vbm=True, include_wholebrain_sbm=True, save=False, onlydeep=False):
    """
    runs the meta model using scores of classifiers trained and tested with the ypred_voxelwise_sbm_te features : VBM ROI, VBM voxelwise, SBM ROI
    the corresponding classifiers are an support vector machine with RBF kernel, a 5-DE TL densenet121, and en elastic net, respectively.
    include_voxelwise_vbm (bool) : whether to include voxelwise VBM classifier (5-DE TL model) in the meta model
    N763 (bool) : whether to use classifiers trained on N763 dataset (no data impuation for SBM), or use the whole 861 subjects' data but therefore using 
                    data imputation for SBM ROI (elastic net model), since not all subjects had SBM preprocessing available.
    save (bool) : if True, save the classification results of the meta model
    onlydeep (bool) : use only DL produces scores in the meta model (scores from VBM voxelwise and SBM wholebrain)
    """
    sites= get_predict_sites_list()

    for n in range(9):
        print("n : ", n)
        pathsbm = ROOT+"results_classif/meta_model/EN_N"+str(n)+"_Destrieux_SBM_ROI_N763.pkl"
        sbm = read_pkl(pathsbm)

        pathvbm = ROOT+"results_classif/meta_model/scores_tr_te_N763_train_size_N"+str(n)+"_vbmroi.pkl"
        vbm = read_pkl(pathvbm)
        # reorder vbm roi train and test scores to match sbm roi order (participant-id wise)
        vbm = reorder_vbm_roi_to_match_sbm_roi_order(sbm, vbm, sites)

        if include_voxelwise_vbm:
            dl_tl_5de_tr, _ = get_5DE_VBM_voxelwise_763(train=True, test=False, verbose=False)
            dl_tl_5de_te, _ = get_5DE_VBM_voxelwise_763(train=False, test=True, verbose=False)
            
        if include_wholebrain_sbm:
            list_sizes_sbm=[75, 150, 200, 300, 400, 450, 500, 600, 700]
            dl_tl_5de_tr_sbm = read_pkl("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/meta_model_sbm_tr_mean_ypreds_5DETL.pkl")
            dl_tl_5de_te_sbm = read_pkl("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/meta_model_sbm_te_mean_ypreds_5DETL.pkl")
            sbm_vertexwise = read_pkl(f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifSBM/L2LR_N{list_sizes_sbm[n]}_SBM_vertexwise_N763.pkl")
        # print(dl_tl_5de_tr)
        results_dict = {}

        if include_voxelwise_vbm or include_wholebrain_sbm:

            mean_auc_over_all_combinations, mean_bacc_over_all_combinations = [], []
            mean_auc_for_each_site = []
            mean_auc_for_each_site_sbm, mean_auc_for_each_site_vbm , mean_auc_for_each_site_voxelwise_vbm= [], [], []

            for site in sites:
                print(site)
                sbm_current_site = sbm[site]
                vbm_current_site = vbm[site]
                sbm_vertexwise_site = sbm_vertexwise[site]

                if include_voxelwise_vbm :
                    maskvbm_tr = (
                        (dl_tl_5de_tr["site"] == site) &
                        (dl_tl_5de_tr["train_idx"] == n)
                    )
                    maskvbm_te = (
                        (dl_tl_5de_te["site"] == site) &
                        (dl_tl_5de_te["train_idx"] == n)
                    )
                    dl_tl_5de_tr_one_combination = dl_tl_5de_tr[maskvbm_tr].copy()
                    dl_tl_5de_te_one_combination = dl_tl_5de_te[maskvbm_te].copy()
                    ypred_voxelwise_vbm_tr = dl_tl_5de_tr_one_combination["mean_ypred"].values
                    ypred_voxelwise_vbm_te = dl_tl_5de_te_one_combination["mean_ypred"].values
                    ypred_voxelwise_vbm_tr = ypred_voxelwise_vbm_tr[0]
                    ypred_voxelwise_vbm_te = ypred_voxelwise_vbm_te[0]

                if include_wholebrain_sbm:
                    ypred_vertexwise_sbm_tr = sbm_vertexwise_site["y_pred_tr"]
                    ypred_vertexwise_sbm_te = sbm_vertexwise_site["y_pred_te"]
                    # print("ypred SBM te size ",np.shape(ypred_voxelwise_sbm_te),"  ",np.shape(ypred_vertexwise_sbm_tr),ypred_vertexwise_sbm_te)
                    # print("ypred SBM te size ",type(ypred_vertexwise_sbm_te),"  ",type(ypred_vertexwise_sbm_tr))

                if include_wholebrain_sbm and include_voxelwise_vbm: 
                    classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr, \
                                    ypred_voxelwise_vbm_te, ypred_vertexwise_sbm_tr, ypred_vertexwise_sbm_te, onlydeep=onlydeep)
                elif include_voxelwise_vbm and not include_wholebrain_sbm : 
                    classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr, \
                                    ypred_voxelwise_vbm_te)
                elif include_wholebrain_sbm and not include_voxelwise_vbm:
                    classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_sbm_tr=ypred_vertexwise_sbm_tr, \
                                        ypred_voxelwise_sbm_te=ypred_vertexwise_sbm_te)
                        
            # for combination_nb in range(1,253):
            #     # print("combination_nb ",combination_nb)
            #     for site in sites:
            #         sbm_current_site = sbm[site]
            #         vbm_current_site = vbm[site]
            #         if include_voxelwise_vbm :
            #             maskvbm_tr = (
            #                 (dl_tl_5de_tr["site"] == site) &
            #                 (dl_tl_5de_tr["train_idx"] == n)
            #             )
            #             maskvbm_te = (
            #                 (dl_tl_5de_te["site"] == site) &
            #                 (dl_tl_5de_te["train_idx"] == n)
            #             )
            #             dl_tl_5de_tr_one_combination = dl_tl_5de_tr[maskvbm_tr].copy()
            #             dl_tl_5de_te_one_combination = dl_tl_5de_te[maskvbm_te].copy()
            #             ypred_voxelwise_vbm_tr = dl_tl_5de_tr_one_combination["mean_ypred"].values
            #             ypred_voxelwise_vbm_te = dl_tl_5de_te_one_combination["mean_ypred"].values
            #             ypred_voxelwise_vbm_tr = ypred_voxelwise_vbm_tr[0]
            #             ypred_voxelwise_vbm_te = ypred_voxelwise_vbm_te[0]

            #         if include_wholebrain_sbm:
            #             mask_sbm_tr = (
            #                 (dl_tl_5de_tr_sbm["site"] == site) &
            #                 (dl_tl_5de_tr_sbm["train_idx"] == n) &
            #                 (dl_tl_5de_tr_sbm["combination"] == combination_nb)
            #             )
            #             mask_sbm_te = (
            #                 (dl_tl_5de_te_sbm["site"] == site) &
            #                 (dl_tl_5de_te_sbm["train_idx"] == n) &
            #                 (dl_tl_5de_te_sbm["combination"] == combination_nb)
            #             )                    
            #             dl_tl_5de_tr_one_combination_sbm = dl_tl_5de_tr_sbm[mask_sbm_tr].copy()
            #             dl_tl_5de_te_one_combination_sbm = dl_tl_5de_te_sbm[mask_sbm_te].copy()
            #             ypred_voxelwise_sbm_tr = dl_tl_5de_tr_one_combination_sbm["mean_ypred"].values
            #             ypred_voxelwise_sbm_te = dl_tl_5de_te_one_combination_sbm["mean_ypred"].values
            #             ypred_voxelwise_sbm_tr = ypred_voxelwise_sbm_tr[0][site]
            #             ypred_voxelwise_sbm_te = ypred_voxelwise_sbm_te[0][site]
            #             # print("ypred SBM te size ",np.shape(ypred_voxelwise_sbm_te),"  ",np.shape(ypred_voxelwise_sbm_tr),ypred_voxelwise_sbm_te)
            #             # print("ypred SBM te size ",type(ypred_voxelwise_sbm_te),"  ",type(ypred_voxelwise_sbm_tr))

            #         if include_wholebrain_sbm and include_voxelwise_vbm: 
            #             classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr, \
            #                          ypred_voxelwise_vbm_te, ypred_voxelwise_sbm_tr, ypred_voxelwise_sbm_te, combination_nb=combination_nb, onlydeep=onlydeep)
            #         elif include_voxelwise_vbm and not include_wholebrain_sbm : 
            #             classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr, \
            #                          ypred_voxelwise_vbm_te, combination_nb=combination_nb)
            #         elif include_wholebrain_sbm and not include_voxelwise_vbm:
            #             classif_one_site_new(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_sbm_tr=ypred_voxelwise_sbm_tr, \
            #                              ypred_voxelwise_sbm_te=ypred_voxelwise_sbm_te, combination_nb=combination_nb)
                        
                        
            #         if combination_nb==1:
            #             sbm_y_te = sbm_current_site["y_test"]
            #             mean_auc_for_each_site_sbm.append(roc_auc_score(sbm_y_te, sbm_current_site["score test"]))
            #             mean_auc_for_each_site_vbm.append(roc_auc_score(sbm_y_te, vbm_current_site["score test"]))
            #             if include_voxelwise_vbm: 
            #                 mean_auc_for_each_site_voxelwise_vbm.append(roc_auc_score(sbm_y_te, ypred_voxelwise_vbm_te))
            
            # changer incrémentation ici si je fais avec DL et combinaisons: 
            results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                                "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites])}#,\
            #"combination_nb":combination_nb}
            
            mean_over_sites_auc = np.mean([results_dict[site]["roc_auc"] for site in sites])
            mean_over_sites_bacc = np.mean([results_dict[site]["balanced-accuracy"] for site in sites])

            mean_auc_for_each_site.append([results_dict[site]["roc_auc"] for site in sites])

            mean_auc_over_all_combinations.append(mean_over_sites_auc)
            mean_bacc_over_all_combinations.append(mean_over_sites_bacc)

            # print("Mean AUC:", mean_over_sites_auc)
            # print("Mean balanced-accuracy:",mean_over_sites_bacc)

            mean_auc_for_each_site = np.array(mean_auc_for_each_site)
            mean_auc_for_each_site = mean_auc_for_each_site.mean(axis=0)

            # t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_sbm)
            # print("Paired t-test p-value btw sbm roi and meta model :", p_value)

            # t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_vbm)
            # print("Paired t-test p-value btw vbm roi and meta model :", p_value)

            # # t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_voxelwise_vbm)
            # # print("Paired t-test p-value btw vbm voxelwise and meta model :", p_value)

            # t_stat, p_value = ttest_rel(mean_auc_for_each_site_vbm, mean_auc_for_each_site_sbm)
            # print("Paired t-test p-value btw vbm roi and sbm roi :", p_value)

            print("\nMean AUC over all combinations ",np.mean(mean_auc_over_all_combinations))
            print("Mean balanced-accuracy over all combinations ",np.mean(mean_bacc_over_all_combinations))

        else:
            for site in sites:
                sbm_current_site = sbm[site]
                vbm_current_site = vbm[site]
                classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, N763=N763)
            
            results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                                    "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites])}
            print("Mean AUC:", np.mean([results_dict[site]["roc_auc"] for site in sites]))
            print("Mean balanced-accuracy:", np.mean([results_dict[site]["balanced-accuracy"] for site in sites]))

    if save: 
        save_pkl(results_dict, RESULTS_DICT_N763)

"""
meta-model results:

for N861
with only VBM ROI and SBM ROI stacking, at maximum training set size (n=8):
    Mean AUC: 0.74875428667064
    Mean balanced-accuracy: 0.658532719675945

with only VBM ROI and VBM voxelwise stacking, at maximum training set size:
    with combination nb 1:
        Mean AUC: 0.769608740053422
        Mean balanced-accuracy: 0.6858811192263289

    mean metrics over all combinations :
        for training set size index idx = 0 (smallest training set size): 
            mean over all combinations  0.6711759806335891
            mean bacc over all combinations  0.5763141531412084
        idx = 1
            mean auc over all combinations  0.7056827134747773
            mean bacc over all combinations  0.6097440597004866
        idx = 2
            mean auc over all combinations  0.7162431388328078
            mean bacc over all combinations  0.6349343886686944
        idx = 3
            mean auc over all combinations  0.7177566925853534
            mean bacc over all combinations  0.6280751938746638
        idx = 4
            mean auc over all combinations  0.739901340297979
            mean bacc over all combinations  0.6625178325431275
        idx = 5
            mean auc over all combinations  0.7321172171717693
            mean bacc over all combinations  0.6649171554001799
        idx = 6
            mean auc over all combinations  0.7503944273500022
            mean bacc over all combinations  0.6578826944979262
        idx = 7
            mean auc over all combinations  0.7600494612782448
            mean bacc over all combinations  0.6930110413595514
        idx = 8
            mean auc over all combinations  0.769822208692828
            mean bacc over all combinations  0.6842651589873633
            
        with classifier = lm.LogisticRegression(C=0.01, class_weight='balanced', fit_intercept=False)
        almost exact same results with GridSearch

at maximum training set size:
Paired t-test p-value btw sbm roi and vbm voxelwise : 2.1315486314235543e-06
Paired t-test p-value btw vbm roi and vbm voxelwise : 0.0005742875290524048
Paired t-test p-value btw vbm roi and sbm roi : 0.00019532726155631125

with only SBM ROI and VBM voxelwise stacking, at maximum training set size:

with only VBM ROI, SBM ROI, and VBM voxelwise stacking, at maximum training set size:

"""

def get_5DE_VBM_voxelwise_763(train=False, test=True, verbose=False):
    
    list_sites=get_predict_sites_list()
    final_results = {}  # final_results[train_idx][site] --> {"mean_ypred": ...}

    for train_idx in range(9):
        final_results[train_idx] = {}
        for site in list_sites:
            final_results[train_idx][site] = {"preds": [], "ytrue": []}

    # get predictions from 5 runs
    for run, path_current_run in enumerate(PATHS_VBM_763_VOXELWISE):
        print("Run", run, "path:", path_current_run)
        for train_idx in range(9):
            for site in list_sites:
                if test : res = read_pkl(path_current_run + f"/n_{train_idx}/Test_{site}_densenet121_vbm_bipolar_epoch199.pkl")
                if train : res = read_pkl(path_current_run + f"/n_{train_idx}/Train_densenet121_vbm_bipolar_epoch_199_site_{site}.pkl")
                y_pred = np.array(res["y_pred"])
                y_true = np.array(res["y_true"])
                final_results[train_idx][site]["preds"].append(y_pred)
                final_results[train_idx][site]["ytrue"].append(y_true)

    roc_results = []
    # get mean predictions
    for train_idx, site_dict in final_results.items():
        for site, info in site_dict.items():
            preds = np.stack(info["preds"], axis=0)  # shape: (runs, n_samples)
            mean_pred = preds.mean(axis=0)
            y_true = info["ytrue"][0] # y_true is the same across runs, so take the first one
            auc = roc_auc_score(y_true, mean_pred) # get ROC-AUC for 5DE TL 
            final_results[train_idx][site] = {"mean_ypred": mean_pred, "ytrue": y_true, "roc_auc": auc}
            roc_results.append({
                "train_idx": train_idx,
                "site": site,
                "roc_auc": auc
            })

    # convert to a DataFrame
    rows = []
    for train_idx, site_dict in final_results.items():
        for site, info in site_dict.items():
            rows.append({
                "train_idx": train_idx,
                "site": site,
                "mean_ypred": info["mean_ypred"]
            })

    df_summary = pd.DataFrame(rows)
    if verbose: print(df_summary)
    df_roc = pd.DataFrame(roc_results)
    df_roc_means = df_roc.groupby("train_idx")["roc_auc"].mean().reset_index()
    if verbose: print(df_roc_means)
    return df_summary, df_roc


# add Cohen's kappa computation here between SBM and VBM test sets predicted labels (VBM from this script, SBM without imputation)
def main():
    run_meta_model_new()
    quit()
    get_5DE_VBM_voxelwise_763(train=True, test=False, verbose=True)
   
    # run_meta_model(save=False,N763=True,include_wholebrain_sbm=False, include_voxelwise_vbm=True, onlydeep=False)
    # # run_meta_model(save=False,N763=True, include_wholebrain_sbm=False)
    # for run_nb in range(1,6):
    #     path = f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/N763_classif_VBM_TL_run{run_nb}/"
    #     for trainset_size in range(8):
    #     path_trainsetsize = path + f"n_{trainset_size}/densenet121_vbm_bipolar_epoch_199_{site}.pth"
if __name__ == "__main__":
    main()


"""
N763 with VBM roi, VBM voxelwise, SBM roi
n :  8
Mean AUC over all combinations  0.7720243724324233
Mean balanced-accuracy over all combinations  0.6964570826464617

N763 VBM roi, VBM voxelwise, SBM roi, wholebrain SBM (ALL):
Mean AUC over all combinations  0.7704957490407714
Mean balanced-accuracy over all combinations  0.6958824104150501

N763 VBM roi, SBM roi, SBM whole brain:
Mean AUC over all combinations  0.7503019060933989
Mean balanced-accuracy over all combinations  0.6728474157942694

N763 VBM roi, SBM roi:
Mean AUC: 0.7508463829558479
Mean balanced-accuracy: 0.6733604444600765

N763 VBM voxelwise, SBM wholebrain:
Mean AUC over all combinations  0.7398693542447192
Mean balanced-accuracy over all combinations  0.6612499695993135

N763 VBM roi, SBM whole brain:
Mean AUC over all combinations  0.7335915163729018
Mean balanced-accuracy over all combinations  0.684185807432612
same with gridsearch:
Mean AUC over all combinations  0.733633983341492
Mean balanced-accuracy over all combinations  0.6829391259005625

N763 SBM roi, SBM whole brain:
Mean AUC over all combinations  0.692956767313462
Mean balanced-accuracy over all combinations  0.6091045004407979
same with gridsearch : 
Mean AUC over all combinations  0.6931660168880558
Mean balanced-accuracy over all combinations  0.6097370725997685

N763 VBM roi, VBM voxelwise:
Mean AUC over all combinations  0.759136935913429
Mean balanced-accuracy over all combinations  0.6747964200359334
"""


"""
with L2LR scores for SBM vertexwise
Mean AUC over all combinations  0.758028468521053
Mean balanced-accuracy over all combinations  0.6161512821992025

"""