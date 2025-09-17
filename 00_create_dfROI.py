import os, pickle, json, gc, sys, numpy as np, pandas as pd, shap, joblib

from utils import save_pkl, get_participants, get_predict_sites_list, get_classifier, get_scores,\
        get_LOSO_CV_splits_N861, get_LOSO_CV_splits_N763, get_scores_pipeline, create_folder_if_not_exists, save_shap_file

#inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATA_DIR=ROOT+"data/processed/"

#outputs
ROI_VBM_FILE = DATA_DIR+"VBM_ROI_allfolds_280725.csv"
ROI_SBM_FILE_Destrieux = DATA_DIR+"SBM_ROI_Destrieux_allfolds_280725.csv"
ROI_SBM_FILE_Desikan = DATA_DIR+"SBM_ROI_Desikan_allfolds_280725.csv"
ROI_SBM_FILE_Destrieux_TIV_scaled = DATA_DIR+"SBM_ROI_Destrieux_allfolds_TIV_scaled_280725.csv"
ROI_SBM_FILE_Desikan_TIVscaled = DATA_DIR+"SBM_ROI_Desikan_allfolds_TIV_scaled_280725.csv"

def create_df_VBM_roi():
    if not os.path.exists(ROI_VBM_FILE):
        """
        creates a pandas dataframe with all VBM rois
        with columns :
            - participant_id
            - session
            - TIV
            - CSF_Vol
            - GM_Vol
            - WM_Vol
            - columns for the roi values
            - SBMandVBM: True or False (respectively N=763 or N=861)
            - lrn_curv_x0, lrn_curv_x1, ..., lrn_curv_x8 : participants for various training set sizes to
                plot learning curves
            - site
            - age
            - sex
        """

        splitsN763 = get_LOSO_CV_splits_N763()    
        splitsN861 = get_LOSO_CV_splits_N861()    

        # function to get the lists of participant_ids of subjects for each train test size 
        # to plot learning curves
        def get_list_subjects_learning_curve_tr(n, site=None):
            assert n in [100,175,250,350,400,500,600,700,800],"wrong training dataset size!"
            if n!=800: assert site is not None, " when n is not Nmax, the testing site needs to be specified"
            if n == 800: subjects_tr_n = [p for site in get_predict_sites_list() for p in splitsN861[site + "-"+str(n)][0]]
            else: subjects_tr_n = [p for p in splitsN861[site + "-"+str(n)][0]]
            subjects_tr_n = list(set(subjects_tr_n))
            return subjects_tr_n


        # read participants dataframe
        participants = get_participants()
        # get list of participants in case with all participants (N=861)
        participants_all_N861 = list(splitsN861["Baltimore-800"][0])+list(splitsN861["Baltimore-800"][1])
        # same thing for participants with both SBM and VBM (N=763)
        participants_all_N763 = list(splitsN763["Baltimore-700"][0])+list(splitsN763["Baltimore-700"][1])
        msk = list(participants[participants['participant_id'].isin(participants_all_N861)].index)
        participants_VBM = participants.iloc[msk]   
        participants_VBM = participants_VBM.reset_index(drop=True)

        # dataframe of all ROIs for all participants (N=861)
        VBMdf = pd.read_csv(DATA_DIR+"VBMROI_Neuromorphometrics.csv")
        # reorder VBMdf to have rows in the same order as participants_VBM
        VBMdf = VBMdf.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()
        new_dfVBM = VBMdf.copy()
        # adding column of booleans indicating which participants have both preprocessings
        new_dfVBM["SBMandVBM"] = new_dfVBM["participant_id"].isin(participants_all_N763)
        # adding age, sex, site, and diagnosis to df
        new_dfVBM = pd.merge(new_dfVBM, participants_VBM[["participant_id","age","sex","site","dx"]], on ="participant_id")

        # adding column of booleans indicating which participants to have in each learning curve train set
        # (to plot performance at different training set sizes)
        cpt = 0
        for n in [100,175,250,350,400,500,600,700,800]:
            if n!=800: 
                # need to specify site because the training set for different LOSO CV sites will not contain the same tr participants 
                # (even when accounting for different test sites) --> for stratification issues
                for site in get_predict_sites_list():
                    new_dfVBM["lrn_curv_x"+str(cpt)+"_"+site] = new_dfVBM["participant_id"].isin(get_list_subjects_learning_curve_tr(n,site))
            else: new_dfVBM["lrn_curv_x"+str(cpt)] = new_dfVBM["participant_id"].isin(get_list_subjects_learning_curve_tr(n))
            cpt+=1

        # verifications
        assert np.isclose(new_dfVBM["TIV"], 1500.0).all()
        assert new_dfVBM["participant_id"].is_unique, "all participant_ids aren't unique in the df"

        new_dfVBM["sex"] = new_dfVBM["sex"].replace({1: "female", 0: "male"})
        print(new_dfVBM) # print new dataframe

        new_dfVBM.to_csv(ROI_VBM_FILE, index=False)
        print("saved new df for VBM ROI to ", ROI_VBM_FILE)

    else : 
        dfroi = pd.read_csv(ROI_VBM_FILE)
        print(dfroi)

        # infos utiles pour comprendre pourquoi la classif avec modèles normatifs marche pas 
        # mask = dfroi['age'].eq(dfroi['sex'])   # or: df['age'] == df['sex']
        # count_equal = mask.sum()
        # print("number of subjects with same age and sex :",count_equal)
        # print(dfroi[dfroi["sex"]==1]["age"])
        # print(dfroi[dfroi["sex"]==0]["age"])

        # import matplotlib.pyplot as plt

        # for s in dfroi["sex"].dropna().unique():
        #     subset = dfroi.loc[dfroi["sex"] == s, "age"].dropna()

        #     plt.figure()
        #     plt.hist(subset, bins=20, edgecolor="black")
        #     plt.title(f"Age distribution – sex = {s}")
        #     plt.xlabel("Age")
        #     plt.ylabel("Frequency")
        #     plt.tight_layout()
        #     plt.show()


def create_df_SBM_roi(atlas="Destrieux", scale_TIV=False):
    assert atlas in ["Destrieux","Desikan"]

    if not scale_TIV:
        filepath = ROI_SBM_FILE_Destrieux if atlas=="Destrieux" else ROI_SBM_FILE_Desikan
    else: 
        filepath = ROI_SBM_FILE_Destrieux_TIV_scaled if atlas=="Destrieux" else ROI_SBM_FILE_Desikan_TIVscaled

    if not os.path.exists(filepath):
        """
        creates a pandas dataframe with all SBM rois
        with columns :
            - participant_id
            - session
            - TIV
            - columns for the roi values
            - lrn_curv_x0, lrn_curv_x1, ..., lrn_curv_x8 : participants for various training set sizes to
                plot learning curves
            - site
            - age
            - sex
        """

        splitsN763 = get_LOSO_CV_splits_N763()    
        if not os.path.exists(ROI_VBM_FILE): 
            print("need to create VBM ROI file first.")
            quit()

        dfroiVBM = pd.read_csv(ROI_VBM_FILE)
        dfroiVBM = dfroiVBM[dfroiVBM["SBMandVBM"]]

        # read participants dataframe
        participants = get_participants()
        # get list of participants with both SBM and VBM (N=763)
        participants_all_N763 = list(splitsN763["Baltimore-700"][0])+list(splitsN763["Baltimore-700"][1])
        msk = list(participants[participants['participant_id'].isin(participants_all_N763)].index)
        participants_SBM = participants.iloc[msk]   
        participants_SBM = participants_SBM.reset_index(drop=True)

        # dataframe of all ROIs for all participants (N=761)
        SBMdf = pd.read_csv(DATA_DIR+"SBMROI_"+atlas+"_CT_SA_subcortical_N763.csv")
        SBMdf["participant_id"] = SBMdf["participant_id"].str.replace("sub-", "", regex=False)

        list_sites = get_predict_sites_list()
        list_lrn_curv_splits = ["lrn_curv_x"+str(i)+"_"+site for i in range(8) for site in list_sites]
        list_lrn_curv_splits = list_lrn_curv_splits+["lrn_curv_x8"]

        elements_to_add = ["participant_id","session","age","sex","site","dx"]+list_lrn_curv_splits
        SBMdf= pd.merge(SBMdf, dfroiVBM[elements_to_add])
        all_rois = [r for r in list(SBMdf.columns) if r!="TIV" and r not in elements_to_add]
        print(all_rois)
        if scale_TIV:
            target_tiv = 1500.0
            scaling_factor = target_tiv / SBMdf["TIV"]
            SBMdf[all_rois+["TIV"]] = SBMdf[all_rois+["TIV"]].mul(scaling_factor, axis=0)

        # verifications
        if scale_TIV: assert np.isclose(SBMdf["TIV"], 1500.0).all()
        assert SBMdf["participant_id"].is_unique, "all participant_ids aren't unique in the df"

        SBMdf["sex"] = SBMdf["sex"].replace({1: "female", 0: "male"})
        print(SBMdf) # print new dataframe

        SBMdf.to_csv(filepath, index=False)
        print("saved new df for SBM ROI to ", filepath)

    else : 
        dfroi = pd.read_csv(filepath)
        print(dfroi)


def main():
    create_df_SBM_roi(atlas="Desikan", scale_TIV=True)
    create_df_SBM_roi(atlas="Destrieux", scale_TIV=True)
    create_df_SBM_roi(atlas="Desikan", scale_TIV=False)
    create_df_SBM_roi(atlas="Destrieux", scale_TIV=False)

if __name__ == "__main__":
    main()

