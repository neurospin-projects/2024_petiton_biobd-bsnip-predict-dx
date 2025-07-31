import pandas as pd, numpy as np
import sys

sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from utils import get_classifier, get_scores_pipeline, get_predict_sites_list, save_pkl

# inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATA_DIR=ROOT+"data/processed/"
ROI_VBM_FILE = DATA_DIR+"VBM_ROI_allfolds_280725.csv"
ROI_SBM_FILE_Destrieux = DATA_DIR+"SBM_ROI_Destrieux_allfolds_280725.csv"
ROI_SBM_FILE_Desikan = DATA_DIR+"SBM_ROI_Desikan_allfolds_280725.csv"
ROI_SBM_FILE_Destrieux_TIV_scaled = DATA_DIR+"SBM_ROI_Destrieux_allfolds_TIV_scaled_280725.csv"
ROI_SBM_FILE_Desikan_TIVscaled = DATA_DIR+"SBM_ROI_Desikan_allfolds_TIV_scaled_280725.csv"


# outputs
RESULTSFOLDER=ROOT+"results_classif/meta_model/"

def remove_zeros(df, verbose=False):
        columns_with_only_zeros = df.columns[(df == 0).all()]
        if verbose : print("columns with zeros ",columns_with_only_zeros)
        column_indices = [df.columns.get_loc(col) for col in columns_with_only_zeros]
        assert set(columns_with_only_zeros) <= {'lInfLatVen_GM_Vol', 'lOC_GM_Vol', 'lInfLatVen_CSF_Vol', 'lOC_CSF_Vol'}
        df = df.drop(columns=columns_with_only_zeros)
        return df


def classify(dfroi, list_rois, datasize_idx, classif_name="EN", formula_res = "age+sex+site"):
    """
        datasize_idx(int): 0 to 8 included, corresponds to the training dataset size index.
                            index 0 corresponds to roughly 100 subjects per train set, index 1 to about 175 subjects, ...
                            and index 8 corresponds to training with a Leave-One-Site-Out CV scheme using all available data
        classif_name (str): classifier name

        formula_res (str): residualization scheme
    """

    assert formula_res in [None, "age+sex+site","age+sex"]

    sites = dfroi["site"].unique()
    
    results_dict = {}
    for site in sites:
        # Test set = all from test site
        test_df = dfroi[dfroi["site"] == site].copy()

        dfroiVBM = pd.read_csv(ROI_VBM_FILE)

        # Training set = subset defined by datasize_idx for this test site
        if datasize_idx != 8: colname = f"lrn_curv_x{datasize_idx}_{site}"
        else : colname = f"lrn_curv_x{datasize_idx}"

        train_df = dfroi[dfroi[colname]].copy()
        train_df = train_df[train_df["site"] != site]


        # ----------------------- start data imputation ------------------------------

        train_df_vbm = dfroiVBM[dfroiVBM[colname]].copy()
        train_df_vbm = train_df_vbm[train_df_vbm["site"]!=site]

        df_vbm_763 = dfroiVBM[dfroiVBM["SBMandVBM"]]
        test_df_vbm = df_vbm_763[df_vbm_763["site"] == site].copy()

        # participants with VBM but not SBM preprocessings for this training set
        missing_ids = train_df_vbm.loc[~train_df_vbm['participant_id'].isin(train_df['participant_id'])]


        lrn_cols = [c for c in train_df.columns if c.startswith("lrn_curv")]

        # means per diagnosis using labels of participants with SBM data
        means_by_dx = train_df.groupby('dx')[list_rois].mean()

        # build new rows for missing participants
        new_rows = []
        for _, row in missing_ids.iterrows():
            dx_value = row['dx']
            # Use mean ROI values for the same dx group
            mean_values = means_by_dx.loc[dx_value]
            
            new_row = {
                'participant_id': row['participant_id'],
                'dx': dx_value
            }

            # Add ROI columns (imputation by dx mean)
            for roi in list_rois:
                new_row[roi] = mean_values[roi]

            # Add session, age, sex, site directly from vbm train df
            for col in ["session", "age", "sex", "site"]:
                new_row[col] = row[col]

            # Fill lrn_curv columns from VBM df (as the SBM roi df doesn't have the values for all subjects)
            for col in lrn_cols:
                new_row[col] = row[col]
                    
            new_rows.append(new_row)

        # convert new rows to DataFrame
        new_df = pd.DataFrame(new_rows)

        # append to train_df 
        train_df_updated = pd.concat([train_df, new_df], ignore_index=True)

        # drop TIV column (since it has NaNs now, the TIV of SBM isn't the same as VBM)
        train_df_updated = train_df_updated.drop(columns=["TIV"])

        train_df_updated = train_df_updated.set_index("participant_id")
        train_df_updated = train_df_updated.loc[train_df_vbm["participant_id"]]
        train_df_updated = train_df_updated.reset_index()

        # Optional: Check shape and order
        print(f"Original train_df shape: {train_df.shape}, Updated df shape: {train_df_updated.shape}")
        # train_df_updated reordered to match train_df_vbm participant order

        train_df = train_df_updated

        """
        # Check for NaNs
        nan_summary = train_df.isna().sum()
        if nan_summary.sum() > 0:
            print("⚠️ NaNs detected in training data:")
            print(nan_summary[nan_summary > 0])
        else:
            print("✅ No NaNs in training data")
        """
        
        # ----------------------- end data imputation ------------------------------

        # make sure the order of the test set values are the same for both dataframes
        test_df = test_df.set_index("participant_id")
        test_df = test_df.loc[test_df_vbm["participant_id"]]
        test_df = test_df.reset_index()

        # combined df for residualization
        combined_df = pd.concat([train_df, test_df], axis=0)

        # Features and labels
        X_train = train_df[list_rois].values
        y_train = train_df["dx"].values
        X_test = test_df[list_rois].values
        y_test = test_df["dx"].values
        participants_train = train_df["participant_id"].values
        participants_test = test_df["participant_id"].values
        print("Xtrain , Xtest shapes ",np.shape(X_train), "  ",np.shape(X_test))

        if formula_res:
            residualizer = Residualizer(data=combined_df[["age","sex","site","dx"]], formula_res=formula_res, formula_full=formula_res+"+dx")
            Zres = residualizer.get_design_mat(combined_df[["age","sex","site","dx"]])

            residualizer.fit(X_train, Zres[:len(X_train)])
            X_train = residualizer.transform(X_train, Zres[:len(X_train)])
            X_test = residualizer.transform(X_test, Zres[len(X_train):])

        # Pipeline
        pipe = make_pipeline(
            StandardScaler(),
            get_classifier(classif_name)
        )
        
        # Fit on train, predict on test
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # get classification scores for current classifier
        score_test = get_scores_pipeline(pipe, X_test)
        score_train = get_scores_pipeline(pipe, X_train)

        roc_auc = roc_auc_score(y_test, score_test)
        bacc = balanced_accuracy_score(y_test, y_pred)

        results_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc,"score test": score_test,"score train": score_train, \
                                    "participant_ids_te":participants_test, "participant_ids_tr":participants_train,\
                                        "y_train": y_train, "y_test":y_test}

    print("AUC per site:", [results_dict[site]["roc_auc"] for site in sites])
    results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                            "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites])}
    print("Mean AUC:", np.mean([results_dict[site]["roc_auc"] for site in sites]))

    return results_dict

def run_loso_cv(datasize_idx, save=False , formula_res = "age+sex+site"):
    """
        datasize_idx(int): 0 to 8 included, corresponds to the training dataset size index.
                            index 0 corresponds to roughly 100 subjects per train set, index 1 to about 175 subjects, ...
                            and index 8 corresponds to training with a Leave-One-Site-Out CV scheme using all available data
        formula_res (str): residualization scheme

    """

    dfroi = pd.read_csv(ROI_SBM_FILE_Destrieux)

    list_cols = list(dfroi.columns)
    # print(list_cols)

    list_lrn_curv_splits = ["lrn_curv_x"+str(i)+"_"+site for i in range(8) for site in get_predict_sites_list()]
    list_lrn_curv_splits = list_lrn_curv_splits+["lrn_curv_x8"]
    other_elements = ["participant_id","session","age","sex","site","TIV","dx"]+list_lrn_curv_splits
    list_rois_ = [r for r in list_cols if r not in other_elements]
    
    assert len(list_rois_) == 296 + 34 # 74 for each ROI type (area and cortical thickness) for both hemispheres

    results_dict = classify(dfroi, list_rois_, datasize_idx, formula_res=formula_res)
    results_file = RESULTSFOLDER+"EN_N"+str(datasize_idx)+"_Destrieux_SBM_ROI_subcortical_N763_data_imputation.pkl"

    if save:
        print("\nsaving classification results ...")
        save_pkl(results_dict, results_file)

def main():
    for n in [0,1,2,3,4,5,6,7,8]:
        run_loso_cv(n, formula_res="age+sex+site", save=True)
    

if __name__ == "__main__":
    main()
