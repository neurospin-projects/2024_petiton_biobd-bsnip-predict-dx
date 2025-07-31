import numpy as np 
import pandas as pd, pickle, sys, os
from classif_VBMROI import remove_zeros
from utils import create_folder_if_not_exists, get_LOSO_CV_splits_N763, get_LOSO_CV_splits_N861, get_participants, get_predict_sites_list
from PCNtoolkit.pcntoolkit.util.utils import create_bspline_basis
from PCNtoolkit.pcntoolkit.normative import estimate
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer


DESIKAN_LABELS="/neurospin/hc/openBHB/resource/freesurfer_atlas-desikan_labels.txt"
DESTRIEUX_LABELS="/neurospin/hc/openBHB/resource/freesurfer_atlas-destrieux_labels.txt"
NEUROMORPHOMETRICS_LABELS="/neurospin/hc/openBHB/resource/cat12vbm_labels.txt"
OPENBHB_DIRECTORY="/neurospin/hc/openBHB/data/"
OPENBHB_PARTICIPANTS_FILE="/neurospin/hc/openBHB/participants.tsv"
OPENBHB_TE_LABELS = "/neurospin/hc/openBHB/test_site_labels_v1.tsv"
OPENBHB_TR_LABELS = "/neurospin/hc/openBHB/train_site_labels_v1.tsv"
OPENBHB_SUBCORTICAL="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/concatenated_aseg_all_openBHBsites.csv"
DATAFOLDER="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"

def get_roi_labels(preproc="VBM", atlas_SBM="Desikan"):
    """
        VBM / SBM / SBM_subcortical: (bool) whether to use VBM or SBM - preprocessed data, or SBM_subcortical, 
                                        which refers to subcortical ROI from Freesurfer (aseg) instead of CT and SA measures (SBM=True)
        atlas_SBM : (str) either "Desikan" or "Destrieux" (VBM is always Neuromorphometrics atlas)
    """
    assert preproc in ["VBM","SBM"],"preproc has to be VBM or SBM"
    # dict_ = {"TMP temporal pole":"PT planum temporalePo", "TTG transverse temporal gyrus": "PT planum temporaleTraGy"}
    assert atlas_SBM in ["Desikan","Destrieux"],"atlas_SBM has to be Desikan or Destrieux"
    if preproc =="VBM" :
        labels = NEUROMORPHOMETRICS_LABELS
        roilabels = list(np.loadtxt(labels, dtype=str))
    
    if preproc == "SBM":
        if atlas_SBM =="Desikan": labels = DESIKAN_LABELS
        if atlas_SBM =="Destrieux": labels = DESTRIEUX_LABELS
        roilabels = list(np.loadtxt(labels, dtype=str))
        roilabels = [transform_string(region, "thickness") for region in roilabels]+ \
            [transform_string(region, "area") for region in roilabels]
                
    return roilabels

def transform_string(s, feature = "thickness"):
    assert feature in ["thickness", "area"]
    parts = s.split('-')
    return f"{parts[1]}_{parts[0]}_"+feature

def save_dataframe_npy_by_subject_and_ROI(directory= OPENBHB_DIRECTORY, save = True, SBM = False, VBM=False, atlas_SBM="Desikan", verbose=False):
    """
        saved dataframe containing all OpenBHB ROI measures for VBM ROI with the Neuromorphometrics atlas or
        SBM ROI with Desikan or Destrieux atlases
    """

    if SBM : VBM=False
    assert not (VBM and SBM), "you have to pick only one preprocessing!"
    assert atlas_SBM in ["Desikan","Destrieux"]

    # 1) load roi labels
    if VBM :
        labels = NEUROMORPHOMETRICS_LABELS
    if SBM :
        if atlas_SBM =="Desikan":
            labels = DESIKAN_LABELS
        else : labels = DESTRIEUX_LABELS
        channels = np.loadtxt("/neurospin/hc/openBHB/resource/freesurfer_channels.txt", dtype=str)
        channels = list(channels)
        
        print("channels", channels)
        print(len(channels))

    roilabels = np.loadtxt(labels, dtype=str)
    roilabels = list(roilabels)
    if SBM :
        roilabels = [transform_string(region, "thickness") for region in roilabels]+ \
            [transform_string(region, "area") for region in roilabels]
        
    if verbose : print("roilabels ", roilabels, "\n", np.shape(roilabels), type(roilabels))

    # 2) create an empty DataFrame with column names : participant_id + roi labels' names
    df = pd.DataFrame(columns=["participant_id"] + roilabels)
    if verbose: print("roilabels", len(roilabels)) # len = 284 for VBM

    # 3) get list of npy files names
    # note : there are as many files / participants for VBM and Freesurfer ROI in the OpenBHB dataset
    if VBM : 
        npy_files = [file for file in os.listdir(directory) if file.endswith("cat12vbm_desc-gm_ROI.npy")]
        
    if SBM : 
        if atlas_SBM =="Desikan": npy_files = [file for file in os.listdir(directory) if file.endswith("freesurfer_desc-desikan_ROI.npy")]
        if atlas_SBM =="Destrieux": npy_files = [file for file in os.listdir(directory) if file.endswith("freesurfer_desc-destrieux_ROI.npy")]
        
    # 4) fill a list of participant ids with the participant id for each file, 
    # and a list of values with the roi values for each participant. 

    list_participant_id = [] # shape : (<nb of participant>)
    list_values = [] # shape : (<nb of participant>, <nb of roi>)

    # qc = "/neurospin/hc/openBHB/qc.tsv"
    # df_qc = pd.read_csv(qc, delimiter='\t')
    # print(df_qc)

    if SBM : 
        list_values_CT, list_values_area = [], []

    for npy_file in npy_files: 
        # Extract the participant_id from the filename
        participant_id = (npy_file.split("_")[0]).split("-")[1]  
        list_participant_id.append(participant_id)
        npy_data = np.load(os.path.join(directory, npy_file))
        npy_data = list(npy_data.squeeze())
        # print(np.shape(npy_data))

        if SBM : # we add cortical thickness values to the list first for freesurfer ROI
            index_CT = channels.index("average_thickness_mm")
            list_values_CT.append(list(npy_data[index_CT]))

            index_area = channels.index("surface_area_mm^2")
            list_values_area.append(list(npy_data[index_area]))

        if VBM : list_values.append(npy_data)

    if SBM : #for freesurfer, we have to add surface area and volume values 
        print(np.shape(list_values_CT), "list_values_CT")
        print(np.shape(list_values_area), "list_values_area")
        list_values = np.concatenate((np.array(list_values_CT), np.array(list_values_area)), axis=1)

    # 5) we fill the dataframe at column "participant_id" with the list containing the participant ids
    df["participant_id"] = list_participant_id

    # at this point, the dataframe is empty except for participant ids
    columns = list(df.columns)
    if verbose : print("list values :", np.shape(list_values))
    list_values = np.array(list_values)
    if verbose : print("columns ", len(columns), "np.shape(list_values)[0]",np.shape(list_values)[1])
    
    # we check that nb of roi names / roi values fit
    assert len(columns)==np.shape(list_values)[1]+1
    
    # 6) for each roi column of the dataframe (we exclude the participant_id column),
    # we want the values of the corresponding column of "list_values" 
    for i in range(1,len(columns)):
        df[columns[i]] = list_values[:,i-1]

    # dataframe where there's a column for participant_id, and the other columns are the ROI
    # for each OpenBHB subject 
    print(df) 
    #  5376 rows (nb of subjects), 
    # 285 cols Neuromorphometrics/VBM (including participant_id),
    # 137 columns Desikan (including participant_id),
    # 
   
    # dataframe saved to csv
    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : preproc = "_SBM_"+atlas_SBM
    if save : 
        create_folder_if_not_exists(os.path.join(os.getcwd(),"NormativeModeling"))
        create_folder_if_not_exists(os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data"))
        path = os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data/")
        df.to_csv(path+"df_ROI_data_by_subject"+preproc+".csv")
        print("dataframe saved to ", os.path.join(path,"df_ROI_data_by_subject"+preproc+".csv"))

    return df

def get_info_OpenBHB(df_participants_,df_train_,df_test_):
    participants_train_df = pd.merge(df_participants_, df_train_, on='participant_id')
    print("Min and Max ages training dataset : " , np.min(participants_train_df["age"]), "  ",np.max(participants_train_df["age"]))
    print(participants_train_df)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.histplot(participants_train_df["age"], bins=30, kde=True)
    # plt.title('Age Distribution OpenBHB NM training set N=3985')
    # plt.xlabel('Age')
    # plt.ylabel('Frequency')
    # plt.show()

    participants_test_df = pd.merge(df_participants_, df_test_, on='participant_id')
    nbtotal = len(participants_train_df)+len(participants_test_df)
    print("Total N = ",nbtotal)
    # nb subjects train, test
    nbtrain , nbtest = len(participants_train_df), len(participants_test_df)
    print("Train N = ",nbtrain, ", Test N = ",nbtest)
    # mean and std of age for train, test
    print("Train age mean ", np.round(participants_train_df["age"].mean(),2),", age std ",np.round(participants_train_df["age"].std(),2))
    print("Test age mean ", np.round(participants_test_df["age"].mean(),2),", age std ",np.round(participants_test_df["age"].std(),2))
    # % of female subjects train, test
    print("Train set Female N = ",round(100*(len(participants_train_df[participants_train_df["sex"]=="female"])/nbtrain),2), "%")
    print("Test set Female N = ",round(100*(len(participants_test_df[participants_test_df["sex"]=="female"])/nbtest),2), "%")
    # % nb of unique sites
    train_sites, test_sites = participants_train_df["site"].values, participants_test_df["site"].values
    print(set(list(train_sites))==set(list(test_sites)))
    print("Train sites :", np.unique(train_sites))
    print("N =",len(np.unique(train_sites)))
    print("Test sites :", np.unique(test_sites))
    print("N =",len(np.unique(test_sites)))

def get_resp_cov_fromBIOBDBSNIP(VBM=False, SBM=False, atlas_SBM="Destrieux", site="Baltimore"):
    """
        VBM (bool) : using VBM preprocessing
        SBM (bool) : using SBM preprocessing
        atlas_SBM (str) : "Destrieux" or "Desikan", atlas chosen for SBM ROI 
        site (str) : name of acquisition site 
        returns : dataframe of training set ROI and covariates (age and sex) for one LOSO CV fold 
            (the site is corresponds to the site of the subjects of the test fold)
    """
    participants = get_participants()
     
    if SBM:
        splits = get_LOSO_CV_splits_N763()
        participants_all = list(splits["Baltimore-"+str(700)][0])+list(splits["Baltimore-"+str(700)][1]) 
        msk = list(participants[participants['participant_id'].isin(participants_all)].index)
        participants_SBM = participants.iloc[msk]   
        participants_SBM = participants_SBM.reset_index(drop=True)

        SBMdf = pd.read_csv(DATAFOLDER+"SBMROI_"+atlas_SBM+"_CT_SA_subcortical_N763.csv")
        # reorder SBMdf to have rows in the same order as participants_SBM
        SBMdf = SBMdf.set_index('participant_id').reindex(participants_SBM["participant_id"].values).reset_index()
        
        df_tr_ = SBMdf[SBMdf["participant_id"].isin(splits[site+"-"+str(700)][0])] # datasize set to Nmax (for SBM=700)
        df_tr_ = df_tr_.drop(columns=["TIV"])
        participants_SBM_HC = participants_SBM[participants_SBM["dx"]==0]

        assert len(participants_SBM_HC["participant_id"].values)==421,"wrong number of HC in N861 dataset used for SBM ROI"

        df_tr_HC = pd.merge(df_tr_, participants_SBM_HC, on ="participant_id")[list(df_tr_.columns)]
        assert len(df_tr_)>len(df_tr_HC)

        covtrHC = pd.merge(df_tr_HC, participants_SBM_HC, on ="participant_id")[["participant_id","age","sex","site","dx"]]
        assert all(covtrHC["dx"].values)==0," error in dx of HC"
        
        # make sure the order of participants is the same with covtrHC and df_tr_HC
        assert list(df_tr_HC["participant_id"].values)==list(covtrHC["participant_id"].values), \
            "covariates and responses values not aligned: participants' data not in the same order!"
    
    if VBM:
        splits = get_LOSO_CV_splits_N861()
        participants_all = list(splits["Baltimore-"+str(800)][0])+list(splits["Baltimore-"+str(800)][1]) 
        msk = list(participants[participants['participant_id'].isin(participants_all)].index)
        participants_VBM = participants.iloc[msk]   
        participants_VBM = participants_VBM.reset_index(drop=True)

        VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
        # reorder VBMdf to have rows in the same order as participants_VBM
        VBMdf = VBMdf.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()

        df_tr_ = VBMdf[VBMdf["participant_id"].isin(splits[site+"-"+str(800)][0])] # datasize set to Nmax (for VBM=800)

        exclude_elements = ['session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
        df_tr_ = df_tr_.drop(columns=exclude_elements)

        participants_VBM_HC = participants_VBM[participants_VBM["dx"]==0]
        assert len(participants_VBM_HC["participant_id"].values)==481,"wrong number of HC in N861 dataset used for VBM ROI"

        df_tr_HC = pd.merge(df_tr_, participants_VBM_HC, on ="participant_id")[list(df_tr_.columns)]
        assert len(df_tr_)>len(df_tr_HC)
        
        covtrHC = pd.merge(df_tr_HC, participants_VBM_HC, on ="participant_id")[["participant_id","age","sex","site","dx"]]
        assert all(covtrHC["dx"].values)==0," error in dx of HC"
                
        # make sure the order of participants is the same with covtrHC and df_tr_HC
        assert list(df_tr_HC["participant_id"].values)==list(covtrHC["participant_id"].values), \
            "covariates and responses values not aligned: participants' data not in the same order!"
        
    return df_tr_HC, covtrHC

def get_34subcortical_ROI_SBM():
    return ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Thalamus-Proper',\
        'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', \
        '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', \
             'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Left-choroid-plexus', \
                'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Thalamus-Proper', 'Right-Caudate', \
                    'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', \
                        'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle', 'Optic-Chiasm', \
                            'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

def save_responses_and_covariates_ROI_OpenBHB(VBM = False, SBM=False, SBM_subcortical=False, atlas_SBM="Desikan", \
                                              site=None, concatenateOpenBHBBIOBDBSNIP=False):
    """
        VBM / SBM / SBM_subcortical: (bool) whether to use VBM or SBM - preprocessed data, or SBM_subcortical, 
                                        which refers to subcortical ROI from Freesurfer (aseg) instead of CT and SA measures (SBM=True)
        atlas_SBM : (str) either "Desikan" or "Destrieux" (VBM is always Neuromorphometrics atlas)
        addsitespredictBD : (bool) whether to include BIOBD/BSNIP sites
        
        site : (str) (site or fold for LOSO-CV); only given a value if training the WBLR on HC from each classification training set fold, or from HC
                from each training set fold concatenated with OpenBHB subjects (which are all HC since OpenBHB is composed of healthy 'brains' only) 
                --> corresponds to the site of the testing set in LOSO-CV for ONE LOSO fold
        
        concatenateOpenBHBBIOBDBSNIP : (bool) if site is not None, we save the responses and covariates for either : 
                (i) only BIOBDBSNIP HC for a given the LOSO fold/site (tr : HC of all sites except held-out site for testing, te : HC of testing site), or 
                (ii) for the concatenation of BIOBDBSNIP HC for a given fold/site with OpenBHB data 
                    (tr : HC of all sites except held-out site for testing concatenated with OpenBHB train set subjects, 
                    te : HC of testing site concatenated with OpenBHB test set subjects)
                if site is None (iii) we do not consider BIOBD BSNIP to train the NM
                for each site : 
                    if False (i): only BIOBDBSNIP HC of the fold if site is not None
                    if True (ii): concatenate OpenBHB and HC of BIOBDBSNIP chosen fold, 
                    if False and site is None (iii): save response and covariates only for OpenBHB

        Notes : in this function, "train" and "test" refer to training and testing splits for the normative model, whereas "evaluation"
                refers to the data evaluated on the normative model (from the BSNIP/BIOBD cohort), 
                in our goal to classify BD vs HC. The "evaluation" set is completely independent from the 
                normative model; however, evaluating data on the model using the PCNtoolkit requires that it has the same response/covariate
                format, with the same number of covariates, as well as response variables. 
    """

    if SBM_subcortical: SBM=True
    assert not (VBM and SBM), "both preprocessings can't be used simultaneously"
    assert atlas_SBM in ["Desikan", "Destrieux"]
    if site: assert VBM, "SBM not implemented for training of normative model with BIOBDBSNIP HC or BIOBDBSNIP HC concatenated to OpenBHB data"
    
    # 1) read participants' file
    # columns :
    # participant_id  study  sex   age  site diagnosis  tiv    csfv   gmv   wmv  magnetic_field_strength  acquisition_setting
    participants = OPENBHB_PARTICIPANTS_FILE
    df_participants = pd.read_csv(participants, delimiter='\t')

    # 2) read train/test tsv files containing participant ids for each subject in either split.
    file_path_test = OPENBHB_TE_LABELS
    file_path_train = OPENBHB_TR_LABELS
    df_test = pd.read_csv(file_path_test, delimiter='\t') # column names = ['participant_id', 'siteXacq']
    df_train = pd.read_csv(file_path_train, delimiter='\t')

    print("df_participants ",np.shape(df_participants))
    # print(df_participants)
    # print(df_train)
    # print(df_test)
   
    # 3) create dataframes of participant_id, sex, age, and site with participants data for train and test
    participants_train_df = pd.merge(df_participants, df_train, on='participant_id')
    train_df = participants_train_df.loc[:, ['participant_id', 'age', 'sex','site']]

    participants_test_df = pd.merge(df_participants, df_test, on='participant_id')
    test_df = participants_test_df.loc[:, ['participant_id', 'age', 'sex','site']]

    # 4) read file containing ROI brain values for each subject 
    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : 
        preproc = "_SBM_"+atlas_SBM
        allbrain = os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data/df_ROI_data_by_subject"+preproc+".csv")
    if VBM:
        allbrain = os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data/df_ROI_data_by_subject"+preproc+".csv")

    if SBM_subcortical: 
        preproc = "_SBM_"
        SBMdf = pd.read_csv(DATAFOLDER+"SBMROI_"+atlas_SBM+"_CT_SA_subcortical_N763.csv")
        SBMdf=SBMdf.rename(columns={'Left-Thalamus': 'Left-Thalamus-Proper'})
        SBMdf=SBMdf.rename(columns={'Right-Thalamus': 'Right-Thalamus-Proper'})
        subcortical_roinames_BIOBDBSNIP= [roi for roi in list(SBMdf.columns) if not roi.endswith("_area") \
                                          and not roi.endswith("_thickness") and not roi=="TIV"]
        allbrain = OPENBHB_SUBCORTICAL

    brain_data = pd.read_csv(allbrain)

    if SBM_subcortical: brain_data=brain_data[subcortical_roinames_BIOBDBSNIP]
    # print(brain_data)
    
    # 5) create 2 dataframes for test and train containing the brain data for participants of either split
    brain_data_te = pd.merge(brain_data, df_test[['participant_id']], on='participant_id', how='inner')
    if not SBM_subcortical: brain_data_te.drop('Unnamed: 0', axis=1, inplace=True)
    # print(brain_data_te)

    brain_data_tr = pd.merge(brain_data, df_train[['participant_id']], on='participant_id', how='inner')
    if not SBM_subcortical: brain_data_tr.drop('Unnamed: 0', axis=1, inplace=True)
    
    #if residualize on site : generate design matrices for test and train sets of the normative model
    brain_data_with_cov_te = pd.merge(brain_data_te, df_participants[["participant_id","age","sex","site"]], on='participant_id', how='inner')
    assert brain_data_with_cov_te['age'].notna().all(), "NaN values found in age column"
    # replace the strings "male" and "female" in the covariate data to 0 and 1 respectively
    brain_data_with_cov_te["sex"]= brain_data_with_cov_te['sex'].replace({'male': 0, 'female': 1})

    brain_data_with_cov_tr = pd.merge(brain_data_tr, df_participants[["participant_id","age","sex","site"]], on='participant_id', how='inner')
    assert brain_data_with_cov_tr['age'].notna().all(), "NaN values found in age column"
    # replace the strings "male" and "female" in the covariate data to 0 and 1 respectively
    brain_data_with_cov_tr["sex"]= brain_data_with_cov_tr['sex'].replace({'male': 0, 'female': 1})

    if site : 
        df_tr_HC_BIOBDBSNIP, covtrHC_BIOBDBSNIP = get_resp_cov_fromBIOBDBSNIP(VBM=VBM, SBM=SBM, atlas_SBM=atlas_SBM, site=site)
        brain_data_with_cov_tr_BIOBDBSNIP = pd.merge(df_tr_HC_BIOBDBSNIP, covtrHC_BIOBDBSNIP[["participant_id","age","sex","site"]],\
                                                      on='participant_id', how='inner')
        if concatenateOpenBHBBIOBDBSNIP: 
            brain_data_with_cov_tr = pd.concat([brain_data_with_cov_tr, brain_data_with_cov_tr_BIOBDBSNIP], axis=0)
        else:
            brain_data_with_cov_tr = brain_data_with_cov_tr_BIOBDBSNIP
                
    formula_res, formula_full = "site", "site + age + sex"

    residualizer_tr = Residualizer(data=brain_data_with_cov_tr, formula_res=formula_res, formula_full=formula_full)
    Zres_tr = residualizer_tr.get_design_mat(brain_data_with_cov_tr)

    residualizer_te = Residualizer(data=brain_data_with_cov_te, formula_res=formula_res, formula_full=formula_full)
    Zres_te = residualizer_te.get_design_mat(brain_data_with_cov_te)

    # 6) data checks : merging of brain and 'age', 'sex', and 'site' data, drop NaN values (there should be none already)
    brain_data_with_cov_tr = brain_data_with_cov_tr.dropna()
    brain_data_with_cov_te = brain_data_with_cov_te.dropna()

    # 5) separate features and covariates
    if VBM : labels = get_roi_labels("VBM")
    if SBM : 
        if SBM_subcortical:
            labels = get_34subcortical_ROI_SBM()
            assert len(labels)==34 # nb of subcortical ROI
        else : 
            labels = get_roi_labels("SBM", atlas_SBM=atlas_SBM)


    tr_data_features = brain_data_with_cov_tr[labels]
    te_data_features = brain_data_with_cov_te[labels]
    
    print(tr_data_features)
    print(te_data_features)
    
    # residualize ROI on site
    print("residualizing on site the ROI values ...")

    #train
    tr_features = tr_data_features.to_numpy()
    residualizer_tr.fit(tr_features, Zres_tr)
    tr_features = residualizer_tr.transform(tr_features, Zres_tr)
    tr_data_features = pd.DataFrame(tr_features, columns=tr_data_features.columns)

    # test
    te_features = te_data_features.to_numpy()
    residualizer_te.fit(te_features, Zres_te)
    te_features = residualizer_te.transform(te_features, Zres_te)
    te_data_features = pd.DataFrame(te_features, columns=te_data_features.columns)
    
    # covariates : age and sex
    tr_data_covariates = brain_data_with_cov_tr[["age","sex"]]
    te_data_covariates = brain_data_with_cov_te[["age","sex"]]
    
    # 6) change sex value to binary
    print("tr_data_covariates",np.shape(tr_data_covariates)) #(3985, 2)
    print("te_data_covariates",np.shape(te_data_covariates)) #(666, 2)

    print("tr_data_covariates",np.shape(tr_data_covariates)) #(3985, 2)
    print("te_data_covariates",np.shape(te_data_covariates)) #(666, 2)    

    assert np.shape(te_data_features)[0]==np.shape(te_data_covariates)[0], \
        "Not as many subjects in features dataframe than covariates dataframe for testing split"   
    assert np.shape(tr_data_features)[0]==np.shape(tr_data_covariates)[0], \
        "Not as many subjects in features dataframe than covariates dataframe for training split"

    print("Train covariate size is: ", tr_data_covariates.shape) #(3985, 2) (pandas dataframe) 
    print("Test covariate size is: ", te_data_covariates.shape) #(666, 264) 

    print("Train response size is: ", tr_data_features.shape) #(3985, 284)
    print("Test response size is: ", te_data_features.shape) # (666, 284) 

    y_train = tr_data_features
    y_test = te_data_features

    X_train = tr_data_covariates
    X_test = te_data_covariates
    # print(tr_data_covariates["age"].iloc[0])
    # print(type(tr_data_covariates["age"].iloc[0]))

    create_folder_if_not_exists(os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data"))
    path = os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data/")

    if SBM_subcortical: str_sub = "_subcortical"
    else : str_sub = ""

    if site:
        if concatenateOpenBHBBIOBDBSNIP: str_BIOBDBSNIP="_concatenateOpenBHB_BIOBDBSNIP_HC_"+site
        else : str_BIOBDBSNIP= "_onlyBIOBDBSNIP_HC_"+site
    else : str_BIOBDBSNIP=""

    # 6) save to txt files 
    X_train.to_csv(path   + 'cov_tr'+preproc+str_sub+str_BIOBDBSNIP+'.txt', sep = '\t', header=False, index=False)
    #cov_tr_nosite.txt
    y_train.to_csv(path + 'resp_tr'+preproc+str_sub+str_BIOBDBSNIP+'.txt', sep = '\t', header=False, index=False)
    #resp_teoct24_cat12vbm_residualized.txt resp_troct24_cat12vbm_residualized.txt
    X_test.to_csv(path + 'cov_te'+preproc+str_sub+str_BIOBDBSNIP+'.txt', sep = '\t', header=False, index=False)
    y_test.to_csv(path + 'resp_te'+preproc+str_sub+str_BIOBDBSNIP+'.txt', sep = '\t', header=False, index=False)

def create_bspline_basis_openBHB(VBM=False, SBM=False, SBM_subcortical=False, atlas_SBM="Desikan",site=None, concatenateOpenBHBBIOBDBSNIP=False):
    """
        VBM / SBM / SBM_subcortical: (bool) whether to use VBM or SBM - preprocessed data, or SBM_subcortical, 
                                        which refers to subcortical ROI from Freesurfer (aseg) instead of CT and SA measures (SBM=True)
        atlas_SBM : (str) either "Desikan" or "Destrieux" (VBM is always Neuromorphometrics atlas)
        addsitespredictBD : (bool) whether to include BIOBD/BSNIP sites
    """

    if SBM_subcortical: SBM=True
    if SBM : VBM=False
    assert not (VBM and SBM), "you have to pick only one preprocessing!"
    assert atlas_SBM in ["Desikan", "Destrieux"]
    

    if SBM_subcortical: str_sub = "_subcortical"
    else : str_sub = ""
 
    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : preproc = "_SBM_"+atlas_SBM
    if SBM_subcortical: preproc = "_SBM_"

    if site:
        if concatenateOpenBHBBIOBDBSNIP: str_BIOBDBSNIP="_concatenateOpenBHB_BIOBDBSNIP_HC_"+site
        else : str_BIOBDBSNIP= "_onlyBIOBDBSNIP_HC_"+site
        assert VBM, "SBM not implemented for supplementary experiments where the NM is training on BIOBD BSNIP data as well as OpenBHB data"
    else : str_BIOBDBSNIP=""

    df_participants = pd.read_csv(OPENBHB_PARTICIPANTS_FILE, delimiter='\t')
    
    data_dir_ = os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data/")

    # load train & test covariate data matrices
    X_tr = np.loadtxt(os.path.join(data_dir_, 'cov_tr'+preproc+str_sub+str_BIOBDBSNIP+'.txt'))
    X_te = np.loadtxt(os.path.join(data_dir_, 'cov_te'+preproc+str_sub+str_BIOBDBSNIP+'.txt'))

    # add intercept column
    X_tr = np.concatenate((X_tr, np.ones((X_tr.shape[0],1))), axis=1)
    X_te = np.concatenate((X_te, np.ones((X_te.shape[0],1))), axis=1)

    # prendre l age min et max d'ici pour quand site=True
    age_min = round(min(list(df_participants["age"].unique())),1)
    age_max = round(max(list(df_participants["age"].unique())),1)
    print("min and max age :", age_min, age_max) #5.9 88.0
    B = create_bspline_basis(age_min, age_max)

    # create the basis expansion for the covariates
    print('Creating basis expansion ...')

    # create Bspline basis set
    Phi = np.array([B(i) for i in X_tr[:,0]])
    Phis = np.array([B(i) for i in X_te[:,0]])

    X_tr = np.concatenate((X_tr, Phi), axis=1)
    X_te = np.concatenate((X_te, Phis), axis=1)
    print(X_tr)
    print(X_te)

    np.savetxt(os.path.join(data_dir_, 'cov_bspline_tr'+preproc+str_sub+str_BIOBDBSNIP+'.txt'), X_tr)
    np.savetxt(os.path.join(data_dir_, 'cov_bspline_te'+preproc+str_sub+str_BIOBDBSNIP+'.txt'), X_te)
    print("...done.")

def check_equal_values(list1, list2):
    for val1 in list1:
        for val2 in list2:
            if val1 == val2:
                print("The lists have at least one equal value:", val1)
                return True
    return False

def get_list_npy_filenames(list_subject_ids):
    base_pattern = 'sub-{}_preproc-cat12vbm_desc-gm_ROI.npy'
    npy_filenames = []
    # Iterate over the list of integers
    for num in list_subject_ids:
        # Replace '*' with the current integer and append to the result list
        npy_filenames.append(base_pattern.format(num))
    return npy_filenames

def train_normative_model(estimate_ = False, VBM = False, SBM = False, SBM_subcortical=False,\
                           atlas_SBM="Desikan",  model_type="blr", site=None, concatenateOpenBHBBIOBDBSNIP=False):
    
    if SBM_subcortical: SBM=True
    if SBM : VBM=False
    if VBM: 
        SBM=False
        SBM_subcortical=False
    if site:
        if concatenateOpenBHBBIOBDBSNIP: str_BIOBDBSNIP="_concatenateOpenBHB_BIOBDBSNIP_HC_"+site
        else : str_BIOBDBSNIP= "_onlyBIOBDBSNIP_HC_"+site
    else : str_BIOBDBSNIP=""

    assert atlas_SBM in ["Desikan", "Destrieux"]
    assert not (VBM and SBM), "you have to pick only one preprocessing!"

    data_dir = os.path.join(os.getcwd(),"NormativeModeling/OpenBHB_data/")

    # arguments for create_bspline_basis_openBHB
    if VBM : preproc = "_VBM_Neuromorphometrics"
    if SBM : preproc = "_SBM_"+atlas_SBM
    if SBM_subcortical : preproc="_SBM_"

    if SBM_subcortical: str_sub = "_subcortical"
    else : str_sub = ""

    modelname = model_type+preproc+str_BIOBDBSNIP
    if SBM_subcortical: modelname = model_type+"_SBM_"+str_sub
    if model_type =="gpr": modelname=modelname+"_"+model_type+"_withwarping"

    # _defaultoptim when optimizer is the default one for chosen model
    # otherwise, optimizer is powell

    # configure the covariates to use. 
    cov_file_tr = os.path.join(data_dir, "cov_bspline_tr"+preproc+str_sub+str_BIOBDBSNIP+".txt") 
    cov_file_te = os.path.join(data_dir, 'cov_bspline_te'+preproc+str_sub+str_BIOBDBSNIP+'.txt') 

    covtr = np.loadtxt(cov_file_tr,dtype=str)
    covte = np.loadtxt(cov_file_te,dtype=str)

    print("covariates :",np.shape(covtr), np.shape(covte))
    print("covariates :",type(covtr), type(covte))

    # load train & test response files
    resp_file_tr = os.path.join(data_dir, 'resp_tr'+preproc+str_sub+str_BIOBDBSNIP+'.txt') 
    resp_file_te = os.path.join(data_dir, 'resp_te'+preproc+str_sub+str_BIOBDBSNIP+'.txt')

    resptr = np.loadtxt(resp_file_tr,dtype=float)
    respte = np.loadtxt(resp_file_te,dtype=float)
    print("responses :",np.shape(resptr), np.shape(respte), type(resptr), type(respte))
    print(resp_file_tr)
    quit()

    create_folder_if_not_exists(os.path.join(os.getcwd(),"NormativeModeling/models"))
    create_folder_if_not_exists(os.path.join(os.getcwd(),"NormativeModeling/models/"+modelname))
    path_model = os.path.join(os.getcwd(),"NormativeModeling/models/"+modelname)
    print(path_model)    
    
    if estimate_ : 
        os.chdir(path_model)
        estimate(cov_file_tr,
                    resp_file_tr,
                    testresp=resp_file_te,
                    testcov=cov_file_te,
                    alg = model_type,
                    optimizer = 'powell',
                    savemodel=True,
                    saveoutput=True,
                    standardize=False, warp = "WarpSinArcsinh")
        os.chdir("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/scripts/")
    
   
"""
    STEP 1 : create response and covariates files for training of normative model on VBM or SBM ROI
            save_responses_and_covariates_ROI_OpenBHB()
            make sure 'site' variable is None to only select OpenBHB subjects to generate resp and cov files

    STEP 2 : create bspline basis for train and test covariates
            create_bspline_basis_openBHB(VBM=False, SBM=False, SBM_subcortical=False, atlas_SBM="Desikan",site=None, concatenateOpenBHBBIOBDBSNIP=False)
            keep concatenateOpenBHBBIOBDBSNIP=False and site=None when training and testing the normative model on all OpenBHB (no subjects from BIOBD/BSNIP)
            atlas_SBM will only be taken into account if SBM is True. Choose SBM = True or VBM = True depending on the type of ROI used for your NM.
            SBM cortical (area and cortical surface derived from Freesurfer) are treated separately from SBM subcortical measures.

    STEP 3 : train normative model from covariates' bsplines and response variables. 
                set estimate_ to True to estimate the model.
                SBM, VBM, and SBM_subcortical variables are used the same way as in previous functions. 


    Supplementary experiments (on VBM ROI only, since VBM ROI classification outperforms SBM ROI classification, we chose
        to implement these experiments only for VBM ROI)
        to train on concatenated OpenBHB + BIOBD/BSNIP on the HC of the training set for each fold (for Baltimore site here): 
        1) save_responses_and_covariates_ROI_OpenBHB(VBM=True, concatenateOpenBHBBIOBDBSNIP=True, site="Baltimore")
        2) create_bspline_basis_openBHB(VBM=True, site="Baltimore", concatenateOpenBHBBIOBDBSNIP=True)
        3) train_normative_model(estimate_ = True, VBM = True, model_type="blr", site="Baltimore", concatenateOpenBHBBIOBDBSNIP=True)

        to train on BIOBD/BSNIP on the HC of the training set for each fold :
        1) save_responses_and_covariates_ROI_OpenBHB(VBM=True, concatenateOpenBHBBIOBDBSNIP=False, site="Baltimore")
        2) create_bspline_basis_openBHB(VBM=True, site="Baltimore", concatenateOpenBHBBIOBDBSNIP=False)
        3) train_normative_model(estimate_ = True, VBM = True, model_type="blr", site="Baltimore", concatenateOpenBHBBIOBDBSNIP=False)
"""

def main():
    save_responses_and_covariates_ROI_OpenBHB(VBM=True, concatenateOpenBHBBIOBDBSNIP=False, site="galway")
    train_normative_model(estimate_ = True, VBM = True, SBM = False, SBM_subcortical=False,\
                    atlas_SBM="Desikan",  model_type="blr", site="galway", concatenateOpenBHBBIOBDBSNIP=False)

    quit()

    for site in get_predict_sites_list():
        train_normative_model(estimate_ = True, VBM = True, SBM = False, SBM_subcortical=False,\
                            atlas_SBM="Desikan",  model_type="blr", site=site, concatenateOpenBHBBIOBDBSNIP=True)
    for site in get_predict_sites_list():
        save_responses_and_covariates_ROI_OpenBHB(VBM = True, SBM=False, SBM_subcortical=False, atlas_SBM="Destrieux", site=site, concatenateOpenBHBBIOBDBSNIP=True)
        create_bspline_basis_openBHB(VBM=True, SBM=False, SBM_subcortical=False, atlas_SBM="Desikan",site=site, concatenateOpenBHBBIOBDBSNIP=False)
    # get_resp_cov_fromBIOBDBSNIP(VBM=False, SBM=True, SBM_atlas="Destrieux", site="Baltimore")
    # save_responses_and_covariates_ROI_OpenBHB(VBM = False, SBM=False, SBM_subcortical=True, atlas_SBM="Desikan")
    create_bspline_basis_openBHB(VBM=False, SBM=True, SBM_subcortical=True, atlas_SBM="Desikan")
    train_normative_model(estimate_ = True, VBM = False, SBM = True, SBM_subcortical=True,\
                           atlas_SBM="Desikan",  model_type="blr")
    # train_normative_model(estimate_ = True, predict_ = False, VBM = True, SBM = False, SBM_subcortical=False,  model_type="blr")
    # train_normative_model(estimate_ = True, predict_ = False, VBM = False, SBM = True, SBM_subcortical=False,\
    #                         atlas_SBM="Desikan",  model_type="blr")
    train_normative_model(estimate_ = True, predict_ = False, VBM = False, SBM = True, SBM_subcortical=False,\
                            atlas_SBM="Destrieux",  model_type="blr")
    # save_dataframe_npy_by_subject_and_ROI(SBM=True, verbose=True, atlas_SBM="Destrieux")
    # save_responses_and_covariates_ROI_OpenBHB(SBM=True, atlas_SBM="Destrieux")
    # save_responses_and_covariates_ROI_OpenBHB(VBM=True)
    # create_bspline_basis_openBHB(VBM=True, SBM=False, SBM_subcortical=False)
    # create_bspline_basis_openBHB(VBM=False, SBM=True, SBM_subcortical=False, atlas_SBM="Desikan")
    # create_bspline_basis_openBHB(VBM=False, SBM=True, SBM_subcortical=False, atlas_SBM="Destrieux")

if __name__ == '__main__':
    main()


