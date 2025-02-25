
import os, numpy as np, pandas as pd, sys, pickle
from PCNtoolkit.pcntoolkit.util.utils import create_bspline_basis
from PCNtoolkit.pcntoolkit.normative import predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from classif_VBMROI import remove_zeros
from matplotlib.lines import Line2D
from df_results_and_learning_curves import get_list_mean_and_std, plot_line
from utils import get_participants, get_predict_sites_list,save_pkl, has_nan, has_inf, has_zero, has_zeros_col, \
        get_LOSO_CV_splits_N861, get_LOSO_CV_splits_N763, create_folder_if_not_exists, get_classifier, get_scores, read_pkl
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

DATAFOLDER="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
NEUROMORPHOMETRICS_LABELS="/neurospin/hc/openBHB/resource/cat12vbm_labels.txt"
PATH_ALL_CLASSIF_DF = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/all_classification_results_dataframe.pkl"
PATH_ZSCORES_CLASSIF_DF = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/zscores_classification_results_dataframe.csv"

def get_roi_labels():
    roilabels = np.loadtxt(NEUROMORPHOMETRICS_LABELS, dtype=str)
    return list(roilabels)

def check_diagnosis(merged_cov, labels):
    if 'diagnosis' in merged_cov.columns: 
        diag = merged_cov[["diagnosis"]].copy()
        assert diag['diagnosis'].isin(['bipolar disorder', 'control', 'psychotic bipolar disorder']).all()
        mapping = {'control': 0, 'bipolar disorder': 1, 'psychotic bipolar disorder': 1}
        diag['diagnosis'] = diag['diagnosis'].replace(mapping)
        assert diag['diagnosis'].isin([0,1]).all()
        diag = diag['diagnosis'].values
        assert np.array_equal(diag, labels),"The diagnosis array from get_labels_tr_te_predictBD isn't what it should be!"

def get_VBM_data(datasize, N763=True):
    if N763: 
        splits = get_LOSO_CV_splits_N763()    
        assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"
    else: 
        splits = get_LOSO_CV_splits_N861()    
        assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"

    # read participants dataframe
    participants = get_participants()
    # prepare residualizer for residualization on site only for covariates and response data
    formula_res, formula_full = "site", "site + age + sex + dx"
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
    
    VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    VBMdf["participant_id"] = VBMdf["participant_id"].str.removeprefix("sub-") 
    # reorder VBMdf to have rows in the same order as participants_VBM
    VBMdf = VBMdf.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()

    return participants_VBM, VBMdf,residualizer, Zres, splits

def get_VBM_data_for_current_site(site, datasize, participants_VBM, VBMdf, residualizer, Zres, splits, return_labels=False):
    # get training and testing ROI dataframes (contains participant_id + TIV in addition to 330 ROIs)
    df_tr_ = VBMdf[VBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][0])]
    df_te_ = VBMdf[VBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][1])]
            
    y_train = pd.merge(df_tr_, participants_VBM, on ="participant_id")["dx"].values
    y_test = pd.merge(df_te_, participants_VBM, on ="participant_id")["dx"].values
    covariates_tr = pd.merge(df_tr_, participants_VBM, on ="participant_id")[["age","sex"]]
    covariates_te = pd.merge(df_te_, participants_VBM, on ="participant_id")[["age","sex"]]
    
    # find index in participants df of the train and test subjects for the current LOSO CV site and train data size
    train = list(participants_VBM.index[participants_VBM['participant_id'].isin(splits[site+"-"+str(datasize)][0])])
    test = list(participants_VBM.index[participants_VBM['participant_id'].isin(splits[site+"-"+str(datasize)][1])])
    
    assert list(y_train)==list(participants_VBM.iloc[train]["dx"].values)
    assert list(y_test)==list(participants_VBM.iloc[test]["dx"].values)
    
    # drop participant_ids , sessions, and global measures 
    # (TIV, total cerebrosplinal fluid, gray matter, and white matter volumes)
    exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
    df_tr_ = df_tr_.drop(columns=exclude_elements)
    df_te_ = df_te_.drop(columns=exclude_elements)

    # get ROI names in a list
    assert list(df_tr_.columns) == list(df_te_.columns)
    roi_names = list(df_te_.columns)

    df_tr_ = remove_zeros(df_tr_)
    df_te_ = remove_zeros(df_te_) 

    #remove zeros before creating np arrays of X_train and X_test as the PCN toolkit won't create zscores for the ROI
    # that contain only zeros anyways 
    # (Neuromorphometrics ROI with only zeros : ['lInfLatVen_GM_Vol', 'lOC_GM_Vol', 'lInfLatVen_CSF_Vol', 'lOC_CSF_Vol'])
    X_train = df_tr_.values
    X_test = df_te_.values

    # fit residualizer
    residualizer.fit(X_train, Zres[train])
    X_train = residualizer.transform(X_train, Zres[train])
    X_test = residualizer.transform(X_test, Zres[test])

    if return_labels: return X_train, X_test, df_tr_, df_te_, covariates_tr, covariates_te, y_train, y_test
    else: return X_train, X_test, df_tr_, df_te_, covariates_tr, covariates_te

def get_SBM_data(datasize, atlas, seven_subcortical_Nunes_replicate=False):
    assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"

    # read splits
    splits = get_LOSO_CV_splits_N763()

    # read participants dataframe
    participants = get_participants()

    # prepare residualizer for residualization on site 
    formula_res, formula_full = "site", "site + age + sex + dx"
    # select the participants for SBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    participants_all = list(splits["Baltimore-"+str(700)][0])+list(splits["Baltimore-"+str(700)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_SBM = participants.iloc[msk]   
    participants_SBM = participants_SBM.reset_index(drop=True)

    residualizer = Residualizer(data=participants_SBM, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants_SBM)
    
    # seven_subcortical_Nunes_replicate changes the file name to SBMROI_<atlas>_CT_SA_subcortical_N763.csv to get 34 subcortical ROIs instead of the 
    # same 7 subcortical ROIs as in Nunes et al.
    if seven_subcortical_Nunes_replicate: str_7ROI = "_7ROIsub"
    else : str_7ROI = ""

    SBMdf = pd.read_csv(DATAFOLDER+"SBMROI_"+atlas+"_CT_SA_subcortical"+str_7ROI+"_N763.csv")
    SBMdf["participant_id"] = SBMdf["participant_id"].str.removeprefix("sub-")

    # reorder SBMdf to have rows in the same order as participants_SBM
    SBMdf = SBMdf.set_index('participant_id').reindex(participants_SBM["participant_id"].values).reset_index()

    return participants_SBM, SBMdf, residualizer, Zres, splits

def get_SBM_data_for_current_site(site, datasize, participants_SBM, SBMdf, residualizer, Zres, splits, \
                                  SBM_subcortical=False, atlas="Destrieux", verbose=False, return_labels=False):
    assert atlas in  ["Destrieux","Desikan"], "wrong atlas name!"

    df_tr_ = SBMdf[SBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][0])]
    df_te_ = SBMdf[SBMdf["participant_id"].isin(splits[site+"-"+str(datasize)][1])]
    y_train = pd.merge(df_tr_, participants_SBM, on ="participant_id")["dx"].values
    y_test = pd.merge(df_te_, participants_SBM, on ="participant_id")["dx"].values
    covariates_tr = pd.merge(df_tr_, participants_SBM, on ="participant_id")[["age","sex"]]
    covariates_te = pd.merge(df_te_, participants_SBM, on ="participant_id")[["age","sex"]]
    
    # find index in participants df of the train and test subjects for the current LOSO CV site and train data size
    train = list(participants_SBM.index[participants_SBM['participant_id'].isin(splits[site+"-"+str(datasize)][0])])
    test = list(participants_SBM.index[participants_SBM['participant_id'].isin(splits[site+"-"+str(datasize)][1])])

    assert list(y_train)==list(participants_SBM.iloc[train]["dx"].values)
    assert list(y_test)==list(participants_SBM.iloc[test]["dx"].values)
  
    df_tr_ = df_tr_.drop(columns=["participant_id", "TIV"])
    df_te_ = df_te_.drop(columns=["participant_id", "TIV"])

    X_train = df_tr_.values
    X_test = df_te_.values

    # get ROI names in a list
    assert list(df_tr_.columns) == list(df_te_.columns)
    roi_names = list(df_te_.columns)

    if not SBM_subcortical: 
        roi_names = [roi for roi in roi_names if roi.endswith("_thickness") or roi.endswith("_area")]
        if atlas=="Destrieux": assert len(roi_names) == 296 # 74 for each ROI type (area and cortical thickness) for both hemispheres
        if atlas=="Desikan": assert len(roi_names) == 136 # 34 for each ROI type (area and cortical thickness) for both hemispheres
        X_train = df_tr_[roi_names].values
        X_test = df_te_[roi_names].values
        df_tr_=df_tr_[roi_names]
        df_te_=df_te_[roi_names]
    else : 
        roi_names= [roi for roi in roi_names if not roi.endswith("_thickness") and not roi.endswith("_area")]
        assert len(roi_names)==34
        X_train = df_tr_[roi_names].values
        X_test = df_te_[roi_names].values
        df_tr_=df_tr_[roi_names]
        df_te_=df_te_[roi_names]

    if verbose:
        print("shape train: ",np.shape(X_train), ", test: ", np.shape(X_test))
        print("type train: ",type(X_train), ", test: ", type(X_test))
        print("roi_names : ", np.shape(roi_names), type(roi_names), " len ", len(roi_names))

    assert not has_nan(X_train) and not has_nan(X_test)

    # for SBM subcortical ROI, the 5th ventricle has values equal to zero for some sites
    if has_zeros_col(X_test) : 
        print("X_test has zeros")
        zero_columns_te = df_te_.columns[(df_te_ == 0).all()]
        # print(df_te_[zero_columns_te])
        if zero_columns_te.tolist() != ['5th-Ventricle']: 
            print(df_te_[zero_columns_te])
            quit()

    if has_zeros_col(X_train) : 
        print("X_train has zeros")
        zero_columns_tr = df_tr_.columns[(df_tr_ == 0).all()]
        # print(df_tr_[zero_columns_tr])
        if zero_columns_tr.tolist() != ['5th-Ventricle']: 
            print(df_te_[zero_columns_tr])
            quit()
    
    # fit residualizer
    residualizer.fit(X_train, Zres[train])
    X_train = residualizer.transform(X_train, Zres[train])
    X_test = residualizer.transform(X_test, Zres[test])
   

    # df_tr_, df_te_ are dataframes of train and test data before residualization
    # X_train and X_test are the np arrays of the ROI data after residualization
    if return_labels: return X_train, X_test, df_tr_, df_te_, covariates_tr, covariates_te, y_train, y_test
    else: return X_train, X_test, df_tr_, df_te_, covariates_tr, covariates_te

def create_resp_cov_BIOBDBSNIP(VBM=False, SBM=False,atlas_SBM="Desikan", SBM_subcortical=False, N763=True, datasize=700,\
                               seven_subcortical_Nunes_replicate=False, verbose=False):
    """
    
    
    """
    if SBM_subcortical: SBM=True
    if N763: assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"
    else: assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"

    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"]

    if VBM: participants_VBM, VBMdf,residualizer, Zres, splits = get_VBM_data(datasize, N763=N763)
    if SBM: participants_SBM, SBMdf, residualizer, Zres, splits = get_SBM_data(datasize, atlas=atlas_SBM, \
                                                                               seven_subcortical_Nunes_replicate=seven_subcortical_Nunes_replicate)
    
    # column names present in Predict BD rois dataframe and not in the OpenBHB roi list

    for site in get_predict_sites_list():
        if verbose : print("site : ", site)
        
        if VBM :
            X_train, X_test, rois_train, rois_test , covariates_tr, covariates_te = \
                get_VBM_data_for_current_site(site, datasize ,participants_VBM, VBMdf, residualizer, Zres, splits, verbose=verbose)
        
        if SBM:
            X_train, X_test, rois_train, rois_test , covariates_tr, covariates_te = \
                get_SBM_data_for_current_site(site, datasize ,participants_SBM, SBMdf, residualizer, Zres, splits, \
                                              SBM_subcortical=SBM_subcortical, atlas=atlas_SBM,verbose=verbose)

        # print(covariates_tr, covariates_te)

        if SBM_subcortical: # only needs to be specified in the case of SBM
            str_sub = "_subcortical"
            if VBM : str_sub = ""
        else : str_sub = ""

        rois_train = pd.DataFrame(X_train, index=rois_train.index, columns=rois_train.columns)
        rois_test = pd.DataFrame(X_test, index=rois_test.index, columns=rois_test.columns)
        
        if VBM : preproc = "_VBM_Neuromorphometrics"
        if SBM : preproc = "_SBM_"+atlas_SBM
        if SBM_subcortical: preproc = "_SBM"

        # save response files and covariates
        path_cohort = os.path.join(os.getcwd(),"NormativeModeling/BIOBDBSNIP")
        create_folder_if_not_exists(path_cohort)
        path_site = os.path.join(os.getcwd(),"NormativeModeling/BIOBDBSNIP/"+site)
        create_folder_if_not_exists(path_site)
        if verbose : print(path_site)
        # print(type(covariates_tr), np.shape(covariates_tr))
        # print(type(rois_train), np.shape(rois_train))
        # print(rois_train, "\n",cov_tr)
        if N763: str_vbm763 = "_N763"
        else : str_vbm763 = "_N861"
 
        covariates_tr.to_csv(os.path.join(path_site, 'cov_tr'+preproc+str_sub+"_"+str(datasize)+str_vbm763+'.txt'), sep = '\t', header=False, index=False)
        rois_train.to_csv(os.path.join(path_site, 'resp_tr'+preproc+str_sub+"_"+str(datasize)+str_vbm763+'.txt'), sep = '\t', header=False, index=False)

        # always the same testing data size so we make sure the file hasn't already been saved
        if not os.path.exists(os.path.join(path_site, 'cov_te'+preproc+str_sub+str_vbm763+".txt")):
            covariates_te.to_csv(os.path.join(path_site, 'cov_te'+preproc+str_sub+str_vbm763+".txt"), sep = '\t', header=False, index=False)
        if not os.path.exists(os.path.join(path_site, 'resp_te'+preproc+str_sub+str_vbm763+".txt")):
            rois_test.to_csv(os.path.join(path_site, 'resp_te'+preproc+str_sub+str_vbm763+".txt"), sep = '\t', header=False, index=False)

        print("...saved")

def create_bspline_basis_BIOBDBSNIP(VBM=False, SBM=False, SBM_subcortical=False, atlas_SBM="Desikan",datasize=700, N763=True):
    if SBM_subcortical: SBM=True
    if N763: assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"
    else: assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"

    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"],"SBM atlas has to be either Desikan or Destrieux"

    # read participants dataframe
    participants = get_participants()

    # read splits
    if N763: 
        splits = get_LOSO_CV_splits_N763()
        Nmax=700
    else : 
        splits = get_LOSO_CV_splits_N861()
        Nmax=800

    # select the participants for SBM ROI (train+test participants of any of the 12 splits)
    # it has to be for max training set size, otherwise it won't cover the whole range of subjects
    participants_all = list(splits["Baltimore-"+str(Nmax)][0])+list(splits["Baltimore-"+str(Nmax)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    df = participants.iloc[msk]   
    df = df.reset_index(drop=True)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.histplot(df['age'], bins=30, kde=True)
    # plt.title('Age Distribution BSNIP BIOBD N=861')
    # plt.xlabel('Age')
    # plt.ylabel('Frequency')
    # plt.show()  
    
    age_min = round(min(list(df["age"].unique())),1)
    age_max = round(max(list(df["age"].unique())),1)

    print("min and max age :", age_min, age_max) # for BIOBD BSNIP, dataset N=863 : 15.0, 79.2 
    # age_min, age_max = 5.9, 88.0 for OpenBHB

    B = create_bspline_basis(age_min, age_max)

    str_sub = ""

    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : preproc = "_SBM_"+atlas_SBM
    if SBM_subcortical: 
        preproc="_SBM"
        str_sub= "_subcortical"

    if N763: str_N763= "_N763"
    else : str_N763 = "_N861"

    # create the basis expansion for the covariates
    print('Creating basis expansion ...')
    for site in get_predict_sites_list():
        print("site : ",site)
        path_cohort = os.path.join(os.getcwd(),"NormativeModeling/BIOBDBSNIP")
        create_folder_if_not_exists(path_cohort)
        data_dir_ = os.path.join(os.getcwd(),"NormativeModeling/BIOBDBSNIP/"+site)
        create_folder_if_not_exists(data_dir_)
        print("data dir for current site: ",data_dir_)

        # load train & test covariate data matrices
        X_tr = np.loadtxt(os.path.join(data_dir_, 'cov_tr'+preproc+str_sub+"_"+str(datasize)+str_N763+'.txt'))
        X_te = np.loadtxt(os.path.join(data_dir_, 'cov_te'+preproc+str_sub+str_N763+'.txt'))

        # add intercept column
        X_tr = np.concatenate((X_tr, np.ones((X_tr.shape[0],1))), axis=1)
        X_te = np.concatenate((X_te, np.ones((X_te.shape[0],1))), axis=1)
        # np.savetxt(os.path.join(data_dir_, 'cov_int_tr'+str_addsites+'.txt'), X_tr)
        # np.savetxt(os.path.join(data_dir_, 'cov_int_te'+str_addsites+'.txt'), X_te)

        # create Bspline basis set
        Phi = np.array([B(i) for i in X_tr[:,0]])
        Phis = np.array([B(i) for i in X_te[:,0]])

        X_tr = np.concatenate((X_tr, Phi), axis=1)
        X_te = np.concatenate((X_te, Phis), axis=1)

        np.savetxt(os.path.join(data_dir_, 'cov_bspline_tr'+preproc+str_sub+"_"+str(datasize)+str_N763+'.txt'), X_tr)
        if not os.path.exists(os.path.join(data_dir_, 'cov_bspline_te'+preproc+str_sub+str_N763+'.txt')):
            np.savetxt(os.path.join(data_dir_, 'cov_bspline_te'+preproc+str_sub+str_N763+'.txt'), X_te)
    print("...done.")

def run_predictions_on_LOSO_BIOBDBSNIP_rois(modelname="blr_VBM_Neuromorphometrics",model_type="blr", \
                                    VBM=True, SBM=False, SBM_subcortical=False, atlas_SBM ="Desikan",datasize=700, N763=True):
    """
        evaluate BIOBD BSNIP ROI on normative models trained with VBM ROI, SBM ROI, or SBM surbcortical ROI
    """
    

    if N763: assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"
    else: assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"

    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"], "SBM atlas has to be either Desikan or Destrieux"
 
    if SBM : VBM = False

    if VBM : assert modelname =="blr_VBM_Neuromorphometrics","wrong model name"
    if not SBM_subcortical :
        if SBM and atlas_SBM=="Destrieux": assert modelname == "blr_SBM_Destrieux","wrong model name"
        if SBM and atlas_SBM=="Desikan" :assert modelname == "blr_SBM_Desikan","wrong model name"
    if SBM_subcortical: assert modelname=="blr_SBM_subcortical"

    print("model : ",modelname)

    if SBM_subcortical: str_sub = "_subcortical"
    else : str_sub = ""

    modelpath = os.path.join(os.getcwd(), "NormativeModeling/models/"+modelname+"/Models")

    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : preproc = "_SBM_"+atlas_SBM
    if SBM_subcortical: preproc = "_SBM"

    if N763: str_N763= "_N763"
    else : str_N763 = "_N861"

    respath = os.path.join(os.getcwd(), "NormativeModeling/models/"+modelname+"/LOSO_BIOBDBSNIP")
    create_folder_if_not_exists(respath)

    for site in get_predict_sites_list():
        print("site : ",site)
        respath_ = os.path.join(respath, site)
        create_folder_if_not_exists(respath_)
        print("respath_ ",respath_)
        datapath = os.path.join(os.getcwd(), "NormativeModeling/BIOBDBSNIP/"+site)
        print("datapath ",datapath)

        cov_file_trLOSO= os.path.join(datapath,'cov_bspline_tr'+preproc+str_sub+"_"+str(datasize)+str_N763+'.txt')
        resp_file_trLOSO = os.path.join(datapath, 'resp_tr'+preproc+str_sub+"_"+str(datasize)+str_N763+'.txt')
        cov_file_teLOSO = os.path.join(datapath, 'cov_bspline_te'+preproc+str_sub+str_N763+'.txt')
        # fixed test size so no need for str(datasize)
        resp_file_teLOSO = os.path.join(datapath, 'resp_te'+preproc+str_sub+str_N763+".txt")

        print("cov file tr \n", np.shape(np.loadtxt(cov_file_trLOSO)))
        print("resp file tr \n",np.shape(np.loadtxt(resp_file_trLOSO)))
        print("cov file te \n", np.shape(np.loadtxt(cov_file_teLOSO)))
        print("resp file te \n",np.shape(np.loadtxt(resp_file_teLOSO)))
    
        respath_tr = os.path.join(respath_,"LOSOtrain_"+str(datasize)+str_N763)
        print(respath_tr)
        create_folder_if_not_exists(respath_tr)
        print("respath_tr ",respath_tr)

        if os.path.exists(respath_tr):
            predict(cov_file_trLOSO,
                    alg='blr',
                    respfile=resp_file_trLOSO,
                    model_path= modelpath,
                    save_path = respath_tr)

        respath_te =os.path.join(respath_,"LOSOtest"+"_"+str(datasize)+str_N763)
        create_folder_if_not_exists(respath_te)
        if os.path.exists(respath_te):
            predict(cov_file_teLOSO,
                    alg=model_type,
                    respfile=resp_file_teLOSO,
                    model_path= modelpath,
                    save_path = respath_te)
            
def get_zscore_array(modelname, site, train=False, test=False, datasize=700, N763=True):   
    if N763: str_N763= "_N763"
    else : str_N763 = "_N861"

    datapath = os.path.join(os.getcwd(), "NormativeModeling/models/"+modelname+"/LOSO_BIOBDBSNIP/"+site)

    if train : 
        resp_path = datapath+"/LOSOtrain_"+str(datasize)+str_N763
    if test : 
        resp_path = datapath+"/LOSOtest_"+str(datasize)+str_N763

    zpath = os.path.join(resp_path,"Z_predict.txt")
    zscore = np.loadtxt(zpath)
    print("shape zscores :",np.shape(zscore))
    if has_nan(zscore) : 
        print("zscore has nan")
        zscore[np.isnan(zscore)] = 0
    zscore[np.isinf(zscore)] = 0

    return zscore

def VBM_ROI_names_with_cols_of_zeros(df):
    exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
    df = df.drop(columns=exclude_elements)
    columns_with_only_zeros = df.columns[(df == 0).all()]
    df = df.drop(columns=columns_with_only_zeros)
    print(columns_with_only_zeros)
    column_names = df.columns.tolist()
    return column_names

def remove_roi_extreme(array, roisBD, verbose = True):
    column_names = VBM_ROI_names_with_cols_of_zeros(roisBD)
    df_array = pd.DataFrame(array, columns=column_names)
    listmax = np.array(df_array.max())
    listmin = np.array(df_array.min())
    column_names = np.array(column_names)
    
    indices_max = [i for i, x in enumerate(listmax) if x > 1e15] #1e6
    indices_min = [i for i, x in enumerate(listmin) if x < -1e15]

    if verbose :
        print("ROI VBM with the maximum z-scores :",column_names[indices_max])
        # print(df_array.iloc[:,indices_max])
        # print(df_array.iloc[:,indices_max].mean())
        # print(df_array.iloc[:,indices_max].std())
        print("ROI VBM with the minimum z-scores :",column_names[indices_min])
        # print(df_array.iloc[:,indices_min])
        u = list(column_names[indices_max])+list(column_names[indices_min])
        extremeROI = list(set(u))
        print(extremeROI)
        if not set(u) <= set(['rFusGy_GM_Vol' ,'rMidOccGy_GM_Vol', 'lSupOccGy_GM_Vol' ,'lSupTemGy_GM_Vol','lVenVen_CSF_Vol']):
            print("different !")
            quit()
        print("ROI VBM with extreme z-scores : ",extremeROI)

    # df_array.iloc[:,indices_max] = df_array.iloc[:,indices_max].apply(lambda x: x.where(x >= 1000, 1000))
    # df_array.iloc[:,indices_min] = df_array.iloc[:,indices_min].apply(lambda x: x.where(x <= -1000, -1000))
    if verbose :
        print("ROI VBM with the maximum z-scores :",column_names[indices_max])
        print(df_array.iloc[:,indices_max])

        print("ROI VBM with the minimum z-scores :",column_names[indices_min])
        print(df_array.iloc[:,indices_min])

    # remove ROI for which the z-scores are extreme
    columns_to_keep = [i for i in range(array.shape[1]) if (i not in indices_max) and (i not in indices_min)]
    if verbose : 
        print(len(columns_to_keep))
        print(np.shape(array))
    # Remove columns
    array_modified = array[:, columns_to_keep]
    if verbose :
        print(" Train : min ",np.min(array_modified), ' max ', np.max(array_modified))
        print(np.shape(array_modified))
    return array_modified, columns_to_keep

def zscores_analysis(df):
    threshold = 1.96 #2
    outside_range = (df < -threshold) | (df > threshold)
    outside_count = outside_range.sum()
    outside_proportion = outside_count / len(df)
    outside_proportion_sorted = outside_proportion.sort_values(ascending=False).to_dict()
    # summary_stats = df.describe()
    # print("Counts of z-scores outside [-1.96, 1.96]:\n", outside_count)
    print("\nProportion of z-scores outside [-1.96, 1.96]:\n")
    for k,v in outside_proportion_sorted.items():
        print(k,"  ",v)
    # print("\nSummary statistics:\n", summary_stats)

def run_classification(classif="svm", N763 = True, VBM=True, SBM=False, seven_subcortical_Nunes_replicate=False,\
                       include_subcorticalROI=False, atlas_SBM="Destrieux", datasize = 700, verbose=False):
    
    if N763: assert datasize in [75, 150, 200, 300, 400, 450, 500, 600, 700],"wrong training dataset size!"
    else: assert datasize in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"

    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"], "SBM atlas has to be either Desikan or Destrieux"
 
    results_dict , metrics_dict= {}, {}
    roc_auc_list , bacc_list = [], []

    if VBM: participants_VBM, roisdf,residualizer, Zres, splits = get_VBM_data(datasize, N763=N763)
    if SBM: participants_SBM, roisdf, residualizer, Zres, splits = get_SBM_data(datasize, atlas=atlas_SBM, \
                                                                               seven_subcortical_Nunes_replicate=seven_subcortical_Nunes_replicate)
    for site in get_predict_sites_list():
        if verbose : print("site : ",site)
        if include_subcorticalROI: SBM=True
        if VBM : modelname="blr_VBM_Neuromorphometrics"  
        if SBM : modelname = "blr_SBM_"+atlas_SBM
        

        if VBM :
            X_train, X_test, rois_train, rois_test , covariates_tr, covariates_te, y_tr, y_te = \
                get_VBM_data_for_current_site(site, datasize ,participants_VBM, roisdf, residualizer, Zres, splits,return_labels=True)
        
        if SBM:
            X_train, X_test, rois_train, rois_test , covariates_tr, covariates_te , y_tr, y_te = \
                get_SBM_data_for_current_site(site, datasize ,participants_SBM, roisdf, residualizer, Zres, splits, atlas=atlas_SBM, return_labels=True)

        tr_ = get_zscore_array(modelname, site,train=True, test=False, datasize=datasize, N763=N763)
        te_ = get_zscore_array(modelname, site,train=False, test=True, datasize=datasize, N763=N763)

        # uncomment to look at the ROIs with zscores >1.96
        if VBM:
            vbm_trdf = pd.DataFrame(X_train.tolist(),columns=list(rois_train.columns))
            vbm_tedf = pd.DataFrame(X_test.tolist(),columns=list(rois_test.columns))
            # create dataframe with all zscores
            result = pd.concat([vbm_trdf, vbm_tedf], axis=0)
            result = result.reset_index(drop=True)
            # uncomment for zscores analysis for each roi over all subjects
            y_all = np.concatenate((y_tr, y_te))
            result["dx"]=y_all
            zscores_HC = result[result["dx"]==0]
            zscores_BD = result[result["dx"]==1]
            # zscores_analysis(zscores_HC)
            # zscores_analysis(zscores_BD)
        

        if SBM and include_subcorticalROI:
            model_sub= "blr_SBM_subcortical"
            tr_sub = get_zscore_array(model_sub, site, train=True, test=False, datasize=datasize, N763=N763)
            te_sub = get_zscore_array(model_sub, site, train=False, test=True, datasize=datasize, N763=N763)
            tr_ = np.concatenate((tr_, tr_sub),axis=1)
            te_ = np.concatenate((te_, te_sub),axis=1)
            if atlas_SBM=="Destrieux":
                assert tr_.shape[1]==330 # 296 Destrieux ROI + 34 subcortical ROI
                assert te_.shape[1]==330
                
            # if not include_subcorticalROI and atlas =="Destrieux": assert len(list(df_tr_.columns)) == 444
            # if include_subcorticalROI and atlas =="Destrieux" : assert len(list(df_tr_.columns)) == 444+39

        if verbose:
            print("tr , te ", type(tr_), np.shape(tr_), type(te_), np.shape(te_))
            print(" Train : min ",np.min(tr_), ' max ', np.max(tr_))
            print(" Test : min ",np.min(te_), ' max ', np.max(te_))

        if VBM :
            tr_, columns_to_keep = remove_roi_extreme(tr_, roisdf, verbose = True)
            te_ = te_[:, columns_to_keep]
            if verbose : print(" Test : min ",np.min(te_), ' max ', np.max(te_))

        assert not (has_nan(tr_) or has_inf(tr_) or has_zero(tr_) or has_nan(te_) or has_inf(te_)or has_zero(te_))     

        assert not any(np.all(tr_[:, i] == 0) for i in range(tr_.shape[1])),"te_ has a column with only zeros!"
        # all_zero_columns = [i for i in range(tr_.shape[1]) if np.all(tr_[:, i] == 0)]
        # if any_all_zeros : print(all_zero_columns) 
        assert not any(np.all(te_[:, i] == 0) for i in range(te_.shape[1])),"te_ has a column with only zeros!"
        # all_zero_columns = [i for i in range(te_.shape[1]) if np.all(te_[:, i] == 0)]
        # if any_all_zeros: print(all_zero_columns)
        

        # get classifier
        classifier = get_classifier(classif)

        scaler_ = StandardScaler()
        tr_ = scaler_.fit_transform(tr_)
        te_ = scaler_.transform(te_)

        assert (not (has_nan(tr_) or has_inf(tr_) or has_zero(tr_) or has_nan(te_) or has_inf(te_)or has_zero(te_))  )

        classifier.fit(tr_, y_tr)
        y_pred = classifier.predict(te_)
        score_test = get_scores(classifier, te_)

        # happens only with datasize=300, MLP classifier, and ROI VBM, and only one subject of Baltimore
        while has_inf(score_test): 
            print("score test has inf value for ",classif," and ",datasize)
            contains_neg_inf = np.any(score_test == -np.inf)
            contains_pos_inf = np.any(score_test == np.inf)
            if contains_neg_inf: 
                print("score test has -inf value")
                # Replace -inf with the minimum finite value
                min_finite_value = np.min(score_test[np.isfinite(score_test)])
                score_test[score_test == -np.inf] = min_finite_value
            if contains_pos_inf: 
                print("score test has +inf value")
                # Replace +inf with the minimum finite value
                max_finite_value = np.max(score_test[np.isfinite(score_test)])
                score_test[score_test == np.inf] = max_finite_value      
        
        bacc = balanced_accuracy_score(y_te, y_pred) 
        roc_auc = roc_auc_score(y_te, score_test)
        # print("roc auc ", roc_auc, " accuracy ", accuracy)
        roc_auc_list.append(roc_auc)
        bacc_list.append(bacc)
        results_dict[site]=np.array(score_test)
        metrics_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc}

    if verbose:
        print(classif)
        print("roc auc list mean : ", round(100*np.mean(roc_auc_list),1))
        print("roc auc list between-site std : ", round(100*np.std(roc_auc_list),1))
        print("balanced_accuracy list mean : ", round(100*np.mean(bacc_list),1))
        print("balanced_accuracy list between-site std : ", round(100*np.std(bacc_list),1))

    if N763: str_N763= "_N763"
    else : str_N763 = "_861"
    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : 
        if include_subcorticalROI : preproc = "_SBM_with_subcortical"
        else : preproc = "_SBM_"+atlas_SBM

    create_folder_if_not_exists(os.path.join(os.getcwd(),"NormativeModeling/results_classif"))
    results_file = os.path.join(os.getcwd(),"NormativeModeling/results_classif/zscores"+preproc+"_metrics_"+str(classif)+"_N"+str(datasize)+str_N763+".pkl")
    save_pkl(metrics_dict,results_file )

    # pklfile = "/neurospin/psy_sbox/temp_sara/NormativeModeling/scripts/stacking/z_scores_"+vbm_str+"_"+classif+str_767+".pkl"
    # save_pkl(results_dict, pklfile) 
    # the saved one has a roc auc of 75.4% (5.4) and bacc of 67.3 % (5.6) (EN with svm roi and 767 subjects)

def fill_dataframe(path, df, datasize_list, N763, VBM=False, SBM=False, include_subcorticalROI=False, seven_subcortical_Nunes_replicate=False, atlas_SBM="Destrieux"):
    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"], "SBM atlas has to be either Desikan or Destrieux"
 
    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : 
        if include_subcorticalROI : preproc = "_SBM_with_subcortical"
        else : preproc = "_SBM_"+atlas_SBM
    if N763: str_N763= "_N763"
    else : str_N763 = "_861"
    for classifier in ["L2LR","EN","svm","MLP","xgboost"]:
        for datasize in datasize_list:
            resultsVBM = path+"zscores"+preproc+"_metrics_"+str(classifier)+"_N"+str(datasize)+str_N763+".pkl"
            vbm_data = read_pkl(resultsVBM)

            for site in get_predict_sites_list():
                df["train_size_category"].append(datasize)
                df["test_roc_auc"].append(vbm_data[site]['roc_auc'])
                df["test_balanced_accuracy"].append(vbm_data[site]['balanced-accuracy'])
                df["fold-site"].append(site)
                df["classifier"].append(classifier)

                if VBM : df["feature_type"].append("zscores_VBM_ROI")
                if SBM : 
                    if include_subcorticalROI: 
                        if seven_subcortical_Nunes_replicate: df["feature_type"].append("zscores_SBM_ROI_7subROI")
                        else : df["feature_type"].append("zscores_SBM_ROI")
                    else : df["feature_type"].append("zscores_SBM_ROI_no_subcortical")

                if VBM : df["atlas"].append("Neuromorphometrics")
                if SBM : df["atlas"].append(atlas_SBM)

                if N763 : df["dataset"].append("N763")
                else : df["dataset"].append("N861")

def get_dataframe_zscores(N763=True, include_subcorticalROI=True, seven_subcortical_Nunes_replicate=False, atlas_SBM="Destrieux"):

    results = {"train_size_category":[],"test_roc_auc": [],"test_balanced_accuracy":[],\
                "fold-site": [], "classifier": [], "feature_type":[], "atlas":[],"dataset":[]}
    
    if N763: 
        datasize_list = [75, 150, 200, 300, 400, 450, 500, 600, 700]
    else: 
        datasize_list = [100,175,250,350,400,500,600,700,800]


    path = os.path.join(os.getcwd(),"NormativeModeling/results_classif/")
    print(path)
    
    
    fill_dataframe(path, results, datasize_list, N763, VBM=True, SBM=False, include_subcorticalROI=False)
    fill_dataframe(path, results, datasize_list, N763, VBM=False, SBM=True, include_subcorticalROI=include_subcorticalROI, atlas_SBM=atlas_SBM)
    results=pd.DataFrame(results)
    print(results)
    results.to_csv(PATH_ZSCORES_CLASSIF_DF, index=None)

    return results

def plot_learning_curves(metric="roc_auc", N763=True, SBM=False, VBM=False, include_subcorticalROI=True, \
                         seven_subcortical_Nunes_replicate=False, atlas_SBM="Destrieux"):
    
    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"], "SBM atlas has to be either Desikan or Destrieux"
    assert metric in  ["roc_auc","balanced_accuracy"]

    if N763: 
        N_values = [75, 150, 200, 300, 400, 450, 500, 600, 700]
        str_N763="_N763"
        dataset="N763"
    else: 
        N_values = [100,175,250,350,400,500,600,700,800]
        str_N763="_N861"
        dataset="N861"

    results_zscores = pd.read_csv(PATH_ZSCORES_CLASSIF_DF)
    print(results_zscores)
    results_all = read_pkl(PATH_ALL_CLASSIF_DF)
    print(results_all)

    fig, ax1 = plt.subplots()
    deep_palette = sns.color_palette("deep")
    cpt = 0

    color_dict={
        "EN" : deep_palette[1],
        "xgboost" : deep_palette[2],
        "L2LR" : deep_palette[3],
        "svm" : deep_palette[4],
        "MLP":"#8B4513",
    }

    label_dict={
        "EN" : "EN",
        "xgboost" : "Gradient Boosting",
        "L2LR" : "L2LR",
        "svm" : "SVM-RBF",
        "MLP":"MLP",
    }

    dict_area_under_curve = {"L2LR":{},"EN":{}, "svm":{} ,"MLP":{},"xgboost":{}}

    if SBM:
        list_features = ["SBM_ROI","zscores_SBM_ROI"]
        if not include_subcorticalROI: list_features = ["SBM_ROI_no_subcortical","zscores_SBM_ROI"]
        if seven_subcortical_Nunes_replicate: list_features = ["SBM_ROI_7subROI","zscores_SBM_ROI"]
        preproc="_SBM_"+atlas_SBM
        atlas=atlas_SBM
    if VBM : 
        list_features = ["VBM_ROI","zscores_VBM_ROI"]
        preproc = "_VBM_Neuromorphometrics"
        atlas="Neuromorphometrics"


    for classifier_ in ["L2LR","EN", "svm" ,"MLP","xgboost"] :
        print(classifier_)
        cpt = 0
        diffauc = 0
        for plot_feature in list_features: 
            # set transparency higher for all classifiers except the best-performing one for current feature type
            if VBM:
                if classifier_=="svm": alpha =1
                else:  alpha = 0.3
            if SBM:
                if classifier_=="EN": alpha=1
                else: alpha = 0.3

            if "zscores" not in plot_feature: 
                meansROI, stds, aucROI = get_list_mean_and_std(results_all, classifier_, plot_feature, atlas, metric, dataset = dataset) 
                dict_area_under_curve[classifier_][plot_feature]=aucROI
                plot_line(meansROI, N_values, color_dict, classifier_, label_dict, diffauc, cpt, linestyle_="solid",alpha=alpha)
                print("ROI :",meansROI, "\n",stds,"\n",aucROI)
                means = meansROI

                
            if "zscores" in plot_feature:
                means_zscores, stds, auczscores = get_list_mean_and_std(results_zscores, classifier_, plot_feature, atlas, metric, dataset = dataset) 
                dict_area_under_curve[classifier_][plot_feature]=auczscores
                diffauc = aucROI-auczscores  
                plot_line(means_zscores, N_values, color_dict, classifier_, label_dict, diffauc, cpt, linestyle_="dashed",alpha=alpha) 
                print("zscores :",means_zscores, "\n",stds,"\n",auczscores)
                means = means_zscores

            cpt+=1

        differences=np.array(meansROI)-np.array(means_zscores)
        t_stat, p_value = ttest_1samp(differences, popmean=0)
        print("classifier : ",classifier_)
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")

        print(classifier_,"  ",plot_feature, " mean ",round(100*means[len(means)-1],3), "std ",round(100*stds[len(stds)-1],3))
        means = np.array(means)
        stds= np.array(stds)
        ax1 = plt.gca()
    quit()

    ax1.set_xlabel("training dataset size", fontsize=25)
    if metric=="roc_auc": ax1.set_ylabel("Mean ROC-AUC over LOSO test sites", fontsize=25)
    if metric=="bacc": ax1.set_ylabel("Mean Balanced-Accuracy over LOSO test sites", fontsize=25)

    legend1 = ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=20)

    if VBM : ax1.set_title("VBM ROI vs NM & VBM ROI classification performance", fontsize=30)
    if SBM : ax1.set_title("SBM ROI vs NM & SBM ROI classification performance", fontsize=30)

   # adding the legend for dashed vs straigt lines for zscores vs ROI
    line = Line2D([0], [0], color='grey', linestyle='dashed')
    line.set_linewidth(4) # choose width of grey line in legend
    line_gray = Line2D([0], [0], color='grey', label='Regular Gray Line')
    line_gray.set_linewidth(4)

    print("dict_area_under_curve ",dict_area_under_curve)
    plotfeature1, plotfeature2 = list_features[0], list_features[1]

    if all(len(v) != 0 for v in dict_area_under_curve.values()):
        str_1 = ", AUC : L2LR ("+str(round(dict_area_under_curve['L2LR'][plotfeature1],2))+"), EN ("+str(round(dict_area_under_curve['EN'][plotfeature1],2))+\
            "), svm ("+str(round(dict_area_under_curve['svm'][plotfeature1],2))+"), MLP ("+str(round(dict_area_under_curve['MLP'][plotfeature1],2))+\
                "), GB ("+ str(round(dict_area_under_curve['xgboost'][plotfeature1],2))+")"
        print(str_1)
        str_2 = ", AUC : L2LR ("+str(round(dict_area_under_curve['L2LR'][plotfeature2],2))+"), EN ("+str(round(dict_area_under_curve['EN'][plotfeature2],2))+\
            "), svm ("+str(round(dict_area_under_curve['svm'][plotfeature2],2))+"), MLP ("+str(round(dict_area_under_curve['MLP'][plotfeature2],2))+\
                "), GB ("+ str(round(dict_area_under_curve['xgboost'][plotfeature2],2))+")"
        print(str_2)
    if VBM:
        legend2 = ax1.legend([line_gray, line],  ['VBM'+str_1,'NM VBM'+str_2],loc="upper left", bbox_to_anchor=(0, 0.15), fontsize=20)
    if SBM:
        legend2 = ax1.legend([line_gray, line],  ['SBM'+str_1,'NM SBM'+str_2],loc="upper left", bbox_to_anchor=(0, 0.15), fontsize=20)

    ax1.add_artist(legend1)
    ax1.add_artist(legend2)
    
    print(dict_area_under_curve)
    path_dictdiff = os.getcwd()+"/NormativeModeling/results_classif/differences_AUC_zscores"+preproc+"_"+metric+str_N763+".pkl"
    path_dict_aucs =  os.getcwd()+"/NormativeModeling/results_classif/AUC_zscores"+preproc+"_"+metric+str_N763+".pkl"

    if all(len(v) != 0 for v in dict_area_under_curve.values()):
        diff = {}
        for classif in ["L2LR","EN", "svm" ,"MLP","xgboost"]:
            diff[classif]=dict_area_under_curve[classif][plotfeature1]-dict_area_under_curve[classif][plotfeature2]
        print(diff)

        save_pkl(diff,path_dictdiff)
        save_pkl(dict_area_under_curve,path_dict_aucs)
        
    ax1.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # if saveplot : plt.savefig(os.getcwd()+'/learning_curve_'+feature+'_'+metric+'.svg', format='svg')
    plt.show()

def ttests_comparisons_zscores_ROI(metric="roc_auc",N763=True, SBM=False, VBM=False, include_subcorticalROI=True, \
                         seven_subcortical_Nunes_replicate=False, atlas_SBM="Destrieux", training_set_size=700):
    """
    Compare performances of classifiers when z-scores or ROI are used as features 
    we do the test with H0 being  zscore performance - ROI performance = 0

    in the paper, we used training_set_size = Nmax (700 for N763)
    
    """
    if not N763: assert training_set_size in [100,175,250,350,400,500,600,700,800]
    else: assert training_set_size in [75, 150, 200, 300, 400, 450, 500, 600, 700]
    assert not(SBM and VBM),"both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan","Destrieux"], "SBM atlas has to be either Desikan or Destrieux"
    assert metric in  ["roc_auc","balanced_accuracy"]

    if N763: dataset="N763"
    else: dataset="N861"
    if VBM : 
        feature_type = "VBM_ROI"
        atlas = "Neuromorphometrics"
    if SBM : 
        feature_type = "SBM_ROI"
        atlas=atlas_SBM

    results_zscores = pd.read_csv(PATH_ZSCORES_CLASSIF_DF)
    results_all = read_pkl(PATH_ALL_CLASSIF_DF)

    for classifier in  ["L2LR", "EN","svm","MLP","xgboost"]:
        print("classifier : ",classifier)
        dfROI = results_all[(results_all['classifier'] == classifier) & (results_all['feature_type'] == feature_type) \
                            & (results_all['atlas'] == atlas) & (results_all['train_size_category'] == training_set_size) \
                                & (results_all['dataset'] == dataset)]
        
        dfROI_zscores = results_zscores[(results_zscores['classifier'] == classifier) & (results_zscores['feature_type'] == "zscores_"+feature_type) \
                                        & (results_zscores['atlas'] == atlas) & (results_zscores['train_size_category'] == training_set_size) \
                                            & (results_zscores['dataset'] == dataset)]

        metric_by_siteROI = dfROI["test_"+metric].values
        metric_by_siteROIzscores = dfROI_zscores["test_"+metric].values

        print("mean vbm ",np.mean(metric_by_siteROI))
        print("mean vbm zscores ", np.mean(metric_by_siteROIzscores))
        differences = np.array(metric_by_siteROIzscores) - np.array(metric_by_siteROI)
        print("mean differences (mean over LOSO-CV folds/sites) : ",np.mean(differences),"\ndifferences: ",differences)
        t_stat, p_value = ttest_1samp(differences, popmean=0)
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")
        if p_value<=0.05 and p_value>0.01: print("*")
        if p_value<=0.01 and p_value>0.001: print("**")
        if p_value<=0.001: print("***")
        if p_value>0.05: print("not significant")
        if p_value<=0.05 and np.mean(differences)<0: print("ROI outperform zscores")
        if p_value<=0.05 and np.mean(differences)>0: print("zscores outperform ROI")
        print("\n")


"""
ORDER TO EXECUTE FUNCTIONS 

1. create_resp_cov_BIOBDBSNIP : 
            create response and covariate files for the chosen brain measures (SBM ROI, VBM ROI, etc.) 
            (covariates chosen here are always age and sex, and residualization on sex is 
            performed before saving response measures)
2. create_bspline_basis_BIOBDBSNIP: 
            creates bspline basis depending on minimum and maximum age values of the cohort on covariates data
3. run_predictions_on_LOSO_BIOBDBSNIP_rois:
            evaluates the response and covariates values of BIOBD/BSNIP subjects for CLASSIFICATION train and test splits separately
            using the right normative model (trained using train_WBLR_OpenBHB.py) depending on the chosen feature types
            (SBM ROI, VBM ROI, etc.)
            generates z-scores for all subjects of CLASSIFICATION train/test splits for varying training set sizes in order to generate learning curves
4. run_classification: 
            runs classification using zscores derived from the normative model trained for a chosen feature type (SBM ROI, VBM ROI, ...) instead of 
            training and testing on ROI values directly
            saves the metrics (roc auc and balanced accuracy) in pkl files in NormativeModeling/results_classif folder 
5. get_dataframe_zscores:
            creates a dataframe with the classification results for varying training set sizes using zscores as features (here the default is zscores
            obtained from VBM ROI Neuromorphometrics and SBM ROI with Destrieux and 34 subcortical ROIs)
6. plot_learning_curves:
            plot the learning curves for classification comparing ROC-AUC (or balanced accuracy) values for varying training set sizes
            using ROI as features or deviation scores for either VBM ROI (VBM=True) or SBM ROI (SBM=True)
            default VBM and SBM features are the same as the previous dataframe.
            if you wish to plot the learning curves for SBM ROI with Desikan or with SBM ROI with only 7 subcortical features, the other functions have to be run 
            with these parameters beforehand and the results have to be added to dataframe generated in get_dataframe_zscores
7. ttests_comparisons_zscores_ROI:
            compare prediction accuracy using ROC-AUC or balanced-accuracy at a chosen training set size 
            between ROI features and zscores ("NM & ROI") features.    
            default compares dataset N763 values at maximum training set size (Nmax=700) for Destrieux + 34 subcortical ROI for SBM ROI,
            and VBM ROI with Neuromorphometrics for VBM ROI
"""

def main():
    """ 
    Parameters :
   
    Aim : 
      
    """
    plot_learning_curves(VBM=True)
    quit()
    ttests_comparisons_zscores_ROI(metric = "roc_auc", SBM=True)
    ttests_comparisons_zscores_ROI(metric = "balanced_accuracy", VBM=True)
    ttests_comparisons_zscores_ROI(metric = "balanced_accuracy", SBM=True)
    
    quit()

    """
    for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        create_resp_cov_BIOBDBSNIP(VBM=False, SBM=False,atlas_SBM="Desikan", SBM_subcortical=True, N763=True, datasize=size,\
                                seven_subcortical_Nunes_replicate=False, verbose=True)
        create_bspline_basis_BIOBDBSNIP(VBM=False, SBM=False, SBM_subcortical=True, atlas_SBM="Desikan",datasize=size, N763=True)
        run_predictions_on_LOSO_BIOBDBSNIP_rois(modelname="blr_SBM_subcortical", model_type="blr", \
                                        VBM=False, SBM=True, SBM_subcortical=True, atlas_SBM ="Destrieux",datasize=size, N763=True)
        for classifier in  ["L2LR", "MLP","svm","xgboost","EN"]:
            run_classification(classif=classifier, N763 = True, VBM=False, SBM=True, seven_subcortical_Nunes_replicate=False,\
                        include_subcorticalROI=True, atlas_SBM="Destrieux", datasize = size, verbose=False)
    
    for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        run_predictions_on_LOSO_BIOBDBSNIP_rois(modelname="blr_SBM_subcortical",model_type="blr", \
                                        VBM=False, SBM=True, SBM_subcortical=True, atlas_SBM ="Destrieux",datasize=size, N763=True)

    run_classification(classif="svm", N763 = True, VBM=False, SBM=True, seven_subcortical_Nunes_replicate=False,\
                       include_subcorticalROI=True, atlas_SBM="Destrieux", datasize = 700, verbose=False)

    for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]: # 700
        for classif in ["xgboost"]:
            run_classification(classif=classif, N763 = True, VBM=True, datasize = size, verbose=True)

    for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        run_predictions_on_LOSO_BIOBDBSNIP_rois(modelname="blr_SBM_Destrieux",model_type="blr", \
                                        VBM=False, SBM=True, SBM_subcortical=False, atlas_SBM ="Destrieux",datasize=size, N763=True)
        run_predictions_on_LOSO_BIOBDBSNIP_rois(modelname="blr_SBM_Desikan",model_type="blr", \
                                        VBM=False, SBM=True, SBM_subcortical=False, atlas_SBM ="Desikan",datasize=size, N763=True)
        run_predictions_on_LOSO_BIOBDBSNIP_rois(modelname="blr_VBM_Neuromorphometrics",model_type="blr", \
                                        VBM=True, SBM=False, SBM_subcortical=False, datasize=size, N763=True)
        
    for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        create_bspline_basis_BIOBDBSNIP(VBM=False, SBM=True, atlas_SBM="Destrieux",datasize=size, N763=True)

            for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
        # create_resp_cov_BIOBDBSNIP(VBM=True, SBM=False, include_subcorticalROI=False, N763=True, datasize=size)
        create_resp_cov_BIOBDBSNIP(VBM=False, SBM=True, atlas_SBM="Destrieux", include_subcorticalROI=False, N763=True, datasize=size)
    """

if __name__ == '__main__':
    main()


