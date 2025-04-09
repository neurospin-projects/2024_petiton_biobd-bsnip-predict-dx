import sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# Statmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from classif_VBMROI import remove_zeros
from utils import get_participants, get_LOSO_CV_splits_N861, get_LOSO_CV_splits_N763
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

# inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATAFOLDER=ROOT+"data/processed/"
# output
RESULTS_FEATIMPTCE_AND_STATS_DIR=ROOT+"results_feat_imptce_and_univ_stats/"

def get_scaled_data(res="no_res", VBM=False, SBM=False):
    """
        res : (str) "no_res", "res_age_sex", or "res_age_sex_site" --> residualization applied to data 
        Aim : returns a dataframe of all subjects' ROI values, age, sex, site, and diagnosis, after standard scaling 
        By default, no residualization is applied to the data, but residualization on age and sex, or age, sex, and site
        can be applied.
    
    """
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"
    if VBM : 
        splits = get_LOSO_CV_splits_N861() 
        Nmax = 800  # using maximum training set size for dataset with all VBM-preprocessed subjects, N861 
    if SBM : 
        splits = get_LOSO_CV_splits_N763()
        Nmax = 700 # using maximum training set size for dataset with all SBM-preprocessed subjects, N763

    assert res in ["res_age_sex_site", "res_age_sex", "no_res"],"not the right residualization option for parameter 'res'!"
    # read participants dataframe
    participants = get_participants()
    if VBM : ROIdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    if SBM : ROIdf = pd.read_csv(DATAFOLDER+"SBMROI_Destrieux_CT_SA_subcortical_N763.csv")

    participants_all = list(splits["Baltimore-"+str(Nmax)][0])+list(splits["Baltimore-"+str(Nmax)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_ROI = participants.iloc[msk]   
    participants_ROI = participants_ROI.reset_index(drop=True)

    # reorder ROIdf to have rows in the same order as participants_ROI
    ROIdf = ROIdf.set_index('participant_id').reindex(participants_ROI["participant_id"].values).reset_index()

    if VBM : 
        exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
        ROIdf = remove_zeros(ROIdf, verbose=False)
    if SBM : exclude_elements = ['participant_id', 'TIV']
    
    ROIdf = ROIdf.drop(columns=exclude_elements)

    X_arr = ROIdf.values
    print("X_arr ",X_arr.shape, type(X_arr))
    if VBM : assert X_arr.shape[1]==280, "there should be 280 features for VBM ROI."
    if SBM : assert X_arr.shape[1]==330, "there should be 330 features for SBM ROI with the Destrieux atlas and subcortical features included."

    if res!="no_res":
        if res=="res_age_sex_site": formula = "age + sex + site"
        elif res=="res_age_sex": formula="age + sex"
        residualizer = Residualizer(
                data=participants_ROI,
                formula_res=formula,
                formula_full=formula + " + dx"
            )
        Zres = residualizer.get_design_mat(participants_ROI)
        # fit residualizer
        residualizer.fit(X_arr, Zres)
        X_arr = residualizer.transform(X_arr, Zres)

    # fit scaler
    scaler_ = StandardScaler()
    X_arr = scaler_.fit_transform(X_arr)
    df_X = pd.DataFrame(X_arr , columns = list(ROIdf.columns))
    df_X[["age", "sex", "site", "dx"]] = participants_ROI[["age", "sex", "site", "dx"]]

    return df_X

def transform_df(df):
    # returns dict of correspondences between df column names before and after
    # being modified to fit statsmodel functions
    original_cols = list(df.columns)
    # Replace +- not alowed in statsmodel formulae
    df.columns = df.columns.str.replace('-', '_MINUS_', regex=False) 
    df.columns = df.columns.str.replace('+', '_PLUS_', regex=False) 
    df.rename(columns={"3rd_MINUS_Ventricle": "_3rd_MINUS_Ventricle"}, inplace=True) # only in SBM ROI
    df.rename(columns={"4th_MINUS_Ventricle": "_4th_MINUS_Ventricle"}, inplace=True) # only in SBM ROI
    df.rename(columns={"5th_MINUS_Ventricle": "_5th_MINUS_Ventricle"}, inplace=True) # only in SBM ROI

    new_cols = list(df.columns)
    string_dict = dict(zip(new_cols, original_cols))
    
    return df, string_dict

def perform_tests(res="res_age_sex_site", save=False, SBM=False, VBM=False):
    """
        res (str) "no_res", "res_age_sex" or "res_age_sex_site" : type of residualization applied to ROI before statistical testing
        save (bool) : save the results in pkl file
    """
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"
    assert res in ["res_age_sex_site", "res_age_sex", "no_res"], f"Wrong residualization variable: {res}"  

    df_X = get_scaled_data(res=res, SBM=SBM, VBM=VBM)
    df_X.dx.replace({0:'HC',1:'BD'}, inplace=True)
    print(list(df_X.columns),"\n\n")
    df_X, string_dict = transform_df(df_X)
    
    # Univariate statistics
    stats = list()
    list_rois = [roi for roi in list(df_X.columns) if roi not in  ["age","sex","site","dx"]] 
    if VBM : assert  len(list_rois)==280,f"wrong number of ROI in df! :{len(list_rois)}"
    if SBM : assert  len(list_rois)==330,f"wrong number of ROI in df! :{len(list_rois)}"
    print(list_rois,"\n\n")

    for var_ in list_rois:
        print(var_)
        # Ordinary Least Squares (OLS) regression is performed for each ROI with diagnosis (dx), sex, age, and site as predictors.
        lm_ = smf.ols('%s ~ dx + sex + age + site' % var_, df_X).fit()

        # print(lm_.model.data.param_names) 
        # prints : ['Intercept', 'dx[T.HC]', 'site[T.Boston]', 'site[T.Dallas]', 'site[T.Detroit]', 'site[T.Hartford]', 'site[T.creteil]', 
        # 'site[T.galway]', 'site[T.geneve]', 'site[T.grenoble]', 'site[T.mannheim]', 'site[T.pittsburgh]', 'site[T.udine]', 'sex', 'age']

        #ANOVA Type II Sum of Squares test is applied to check the effect of variables.
        aov_ = sm.stats.anova_lm(lm_, typ=2) 
        stats_ = [var_] +\
        lm_.tvalues[['dx[T.HC]', 'sex', 'age']].tolist() + \
        lm_.pvalues[['dx[T.HC]', 'sex', 'age']].tolist() + \
        [aov_.loc['dx', "F"], aov_.loc['dx', "PR(>F)"],
        aov_.loc['sex', "F"], aov_.loc['sex', "PR(>F)"],
        aov_.loc['age', "F"], aov_.loc['age', "PR(>F)"]]

        stats.append(stats_) # list gets for each roi the t- and p- statistics for each covariate

    # generate final dataframe summarizing all statistics
    cols = ['ROI', 'diag_t', 'sex_t', 'age_t', 'diag_p', 'sex_p', 'age_p',\
          'diag_f', 'diag_p_anova','sex_f_anova', 'sex_p_anova','age_f_anova', 'age_p_anova']
    
    stats = pd.DataFrame(stats, columns=cols)

    # Change back to original variable names
    stats['ROI'] = stats['ROI'].replace(string_dict) # for ROI column of stats df
    df_X.rename(columns=string_dict, inplace=True) # for column names of ROI dataframe df_X
    print(stats)
    print("number of pvalues <0.05 before correction :",np.sum(stats["diag_p"].values < 0.05)) 

    # Benjamini-Hochberg (BH-FDR) correction applied to diag_p
    _, pcor_fdr_bh, _, _  = multipletests(stats.diag_p, method='fdr_bh')
    _, pcor_fdr_bh_anova, _, _  = multipletests(stats.diag_p_anova, method='fdr_bh')

    print("number of pvalues <0.05 after FDR BH correction :",np.sum(pcor_fdr_bh < 0.05)) 

    # Bonferroni correction
    _, pcor_bonf, _, _  = multipletests(stats.diag_p, method='bonferroni')
    _, pcor_bonf_anova, _, _  = multipletests(stats.diag_p_anova, method='bonferroni')

    print("number of pvalues <0.05 after Bonferroni correction :",np.sum(pcor_bonf < 0.05)) # (stricter correction)

    stats['diag_pcor_fdr_bh'] = pcor_fdr_bh
    stats['diag_pcor_bonferroni'] = pcor_bonf
    stats['diag_pcor_fdr_bh_anova'] = pcor_fdr_bh_anova
    stats['diag_pcor_bonferroni_anova'] = pcor_bonf_anova

    print("nb of ROI where diagnosis pvalues from the linear model are <0.05 after Bonferroni correction for multiple tests :",np.sum(stats["diag_pcor_bonferroni"].values<0.05))
    print("nb of ROI where diagnosis pvalues from the ANOVA are <0.05 after Bonferroni correction for multiple tests :",np.sum(stats["diag_pcor_bonferroni_anova"].values<0.05))

    print(stats[["diag_p","diag_p_anova","diag_pcor_bonferroni","diag_pcor_bonferroni_anova"]])

    if save : 
        if VBM : stats.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_"+res+"_VBM_avril25.xlsx", sheet_name='statsuniv_rois_scaled_'+res, index=False)
        if SBM : stats.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_"+res+"_SBM_avril25.xlsx", sheet_name='statsuniv_rois_scaled_'+res, index=False)


"""
USED FOR FEATURE IMPORTANCE -- USING VBM ROI FOR N861 and training datasize at maximum training set size N ~ 800
FOR SBM ROI, using N763 and N ~ 700 

STEP 1: residualize and scale the ROI data for all participants with get_scaled_data()
STEP 2: fit ordinary least squares for each ROI with smf.ols, fitting ROI ~ dx (diag) + age + sex + site
STEP 3: for each ROI regression, compute an ANOVA type II test to find out which regressors (y, age, sex, or site) are 
        significant in the regression
STEP 4: adjust for multiple comparisons with 2 different methods: bonferroni (more stringent), and FDR-Benjamini Hochberg (looser),
        only on the p-values related to y (Li diag)
STEP 5 : save results dataframe to excel file

number of ROI with pvalues<0.05 for diagnosis,
with any type of residualization: before correction for multiple tests:  226, after Bonferroni : 120 ROI (after FDR BH: 222)

"""
def main():
    # perform_tests(res="no_res", save=True)
    # perform_tests(res="res_age_sex", save=True)
    perform_tests(res="res_age_sex_site", save=True, SBM=True)


if __name__ == "__main__":
    main()