from utils import get_predict_sites_list, get_classifier, get_scores, get_LOSO_CV_splits_N861,\
                get_participants, save_pkl

import numpy as np, nibabel, gc, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import sys, argparse

sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer


DATAFOLDER="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
RESULTSFOLDER="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/"

def get_all_data(verbose=False):
    all_participants_BIOBDBSNIP = "/neurospin/psy_sbox/analyses/202104_biobd-bsnip_cat12vbm_predict-dx/biobd-bsnip_cat12vbm_participants.csv"
    all_participants = pd.read_csv(all_participants_BIOBDBSNIP)

    # Mask
    mask_img = nibabel.load(DATAFOLDER+"mni_cerebrum-gm-mask_1.5mm.nii.gz")
    mask_arr = mask_img.get_fdata() != 0
    assert np.sum(mask_arr != 0) == 331695

    imgs_flat_filename = DATAFOLDER+"biobd-bsnip_cat12vbm_mwp1-gs-flat.npy"

    Xim = np.load(imgs_flat_filename, mmap_mode='r')
    if verbose : print("type Xim :",np.shape(Xim), type(Xim))
    assert Xim.shape[1] == np.sum(mask_arr != 0)

    # create dataframe to have the participant_id corresponding to each numpy array of 3D VBM MRI values
    df_Xim_with_participant_names = pd.DataFrame({
        'participant_id': all_participants['participant_id'], 
        'data': [np.array(row) for row in Xim] 
    })

    return df_Xim_with_participant_names

def load_images(site, df_Xim_with_participant_names, datasize = 800):
    """
        site : (str) LOSO-CV site/fold. must be "Baltimore", "Boston", "Dallas",
                 "Detroit", "Hartford", "mannheim", "creteil", "udine", "galway", "pittsburgh", "grenoble", or "geneve".
        df_Xim_with_participant_names : (df) pandas dataframe with 2 columns, "participant_id" and "data", where each row contains
                the participant_id and its corresponding 3D VBM MRI numpy array 
        datasize : (int) approximate size of the training dataset (it varies depending on the LOSO-CV site as the test set is fixed)
                must be a value in list [100,175,250,350,400,500,600,700,800]

    """
    assert site in get_predict_sites_list()," wrong LOSO-CV site name!"
    assert datasize in [100,175,250,350,400,500,600,700,800], "wrong training data size!"

    # read participants dataframe (contains only participants used in our 
    # experiments that passed the VBM quality checks)
    participants = get_participants()
    print("np.shape participants",np.shape(participants))

    splits = get_LOSO_CV_splits_N861()    

    # prepare residualizer for residualization on age, sex, and site 
    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    # select the participants for VBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    participants_all = list(splits["Baltimore-"+str(800)][0])+list(splits["Baltimore-"+str(800)][1])
    print("participants_all len ", len(participants_all))
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_VBM = participants.iloc[msk]   
    participants_VBM = participants_VBM.reset_index(drop=True)

    residualizer = Residualizer(data=participants_VBM, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants_VBM)

    # reorder voxelwiseVBMdf to have rows in the same order as participants_VBM
    voxelwiseVBMdf = df_Xim_with_participant_names.set_index('participant_id').reindex(participants_VBM["participant_id"].values).copy().reset_index()

    # get training and testing ROI dataframes (contains participant_id + TIV in addition to 330 ROIs)
    df_tr_ = voxelwiseVBMdf[voxelwiseVBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][0])]
    df_te_ = voxelwiseVBMdf[voxelwiseVBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][1])]

    y_train = pd.merge(df_tr_, participants_VBM, on ="participant_id")["dx"].values
    y_test = pd.merge(df_te_, participants_VBM, on ="participant_id")["dx"].values
    
    # find index in participants df of the train and test subjects for the current LOSO CV site and train data size
    train = list(participants_VBM.index[participants_VBM['participant_id'].isin(splits[site+"-"+str(datasize)][0])])
    test = list(participants_VBM.index[participants_VBM['participant_id'].isin(splits[site+"-"+str(datasize)][1])])
    
    assert list(y_train)==list(participants_VBM.iloc[train]["dx"].values)
    assert list(y_test)==list(participants_VBM.iloc[test]["dx"].values)
    
    # drop participant_ids
    df_tr_ = df_tr_.drop(columns="participant_id")
    df_te_ = df_te_.drop(columns="participant_id")
            
    X_train = np.vstack(df_tr_["data"].values) 
    X_test = np.vstack(df_te_["data"].values) 
    assert np.shape(X_train)[1]==331695
    assert np.shape(X_test)[1]==331695

    Zres_tr = Zres[train]
    Zres_te = Zres[test]
                     
    gc.collect()

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, \
            "Zres_tr":Zres_tr, "Zres_te": Zres_te, "residualizer":residualizer}




def classif_voxelwise_VBM(classif="svm", datasize=800, save=False):
    assert classif in ["svm", "MLP", "EN", "L2LR", "xgboost"]
    df_all_data = get_all_data()
    
    assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"
    
    print("training dataset approximate size :", datasize)

    roc_auc_list, bacc_list = [],[]
    metrics_dict = dict()

    for site in get_predict_sites_list():   

        # load images for current site and training dataset size
        dict_variables = load_images(site, df_all_data, datasize)
        X_train, X_test = dict_variables["X_train"], dict_variables["X_test"]
        y_train, y_test = dict_variables["y_train"], dict_variables["y_test"]
        Zres_tr, Zres_te = dict_variables["Zres_tr"], dict_variables["Zres_te"]
        residualizer = dict_variables["residualizer"]
        
        print(" site : ", site)
       
        print("Xtrain",np.shape(X_train), "Xtest",np.shape(X_test))
        print("Zres_train",np.shape(Zres_tr), "Zres_test",np.shape(Zres_te))
        print("y_train",np.shape(y_train), "y_test",np.shape(y_test))

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

        # get classification scores for current classifier
        score_test = get_scores(classifier, X_test)        
        
        bacc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, score_test)
        roc_auc_list.append(roc_auc)
        bacc_list.append(bacc)

        metrics_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc}

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
        results_file = RESULTSFOLDER+"classifVoxelWise/"+classif+"_N"+str(datasize)+"_Neuromorphometrics_VBM_ROI_N861.pkl"
        print("\nsaving classification results ...")
        save_pkl(metrics_dict, results_file)


def main():

    parser = argparse.ArgumentParser()

    # choose the size of the training set (Nmax=800) for classification 
    parser.add_argument("--datasize", type=int, choices = [100,175,250,350,400,500,600,700,800])
    # choose the classifier model
    parser.add_argument("--model", type=str, choices = ["svm", "MLP","EN","xgboost","L2LR"])

    keyboard_args = parser.parse_args()

    classif_voxelwise_VBM(keyboard_args.model, datasize = keyboard_args.datasize)

if __name__ == '__main__':
    main()