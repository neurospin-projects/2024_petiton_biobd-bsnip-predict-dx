# 2024_petiton_biobd-bsnip-predict-dx
Prediction of DX (Bipolar Disorder vs Healthy Controls) using Anatomical MRI

## feature importance - feature_imptce_shap_analysis.py

- For **VBM ROI** features, we use SHAP values evaluated during the training of the best-performing model for this type of feature, the **SVM-RBF** (support vector machine with RBF kernel).

- For **SBM ROI** features, we use SHAP values evaluated during the training of the best-performing model for this type of feature, the **EN** (Elastic Net logistic regression).

### preliminary steps
#### run SHAP values with and without random permutation of labels
- To analyze SHAP values for either feature type, you must have run BD vs HC classification and saved the corresponding SHAP values for each LOSO-CV fold.
For **VBM ROI**, this can be done in **classif_VBMROI.py**, and for **SBM ROI**, in **classif_SBMROI.py**. Use maximum training set size for this (N=700 for SBM ROI, N=800 for VBM ROI). (We want to look at feature importance when the models best discriminate between BD and HC subjects.)
- Additionaly, compute shap values for the chosen feature type with random permutations in the same classification scripts, and repeat this step 30 times to get 30 random predictions for all test set subjects of all LOSO-CV folds.
- For **SBM ROI**, run : 
```python
from utils import get_predict_sites_list 
start_time = time.time()
 # for the EN, SHAP values are estimated with a faster method than the SVM-RBF, so SHAP values for all folds 
 # can be computed at once in about 30 min
for onesite_ in get_predict_sites_list():
    for i in range(30):
        print(onesite_)
        # without permutation of labels (only needs to be done once without permutations)
        if i == 0: classif_sbm_ROI(classif = "EN", datasize = 700, save = False, compute_shap=True, random_labels=False, onesite=onesite_) 
        # with permutation of labels
        classif_sbm_ROI(classif = "EN", datasize = 700, save = False, compute_shap=True, random_labels=True, onesite=onesite_) 
        print("site : ",onesite_)
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"The function took {hours}h {minutes}m {seconds}s to run.") 
```
- For **VBM ROI**, run : 
```python
# for random permutations 
from utils import get_predict_sites_list 
start_time = time.time()
for onesite_ in get_predict_sites_list(): # or run each LOSO CV site/fold separately (about 2h computation type by fold)
    for i in range(30):
        print(onesite_)
        classif_vbm_ROI(classif = "svm", datasize = 800, save = False, N763=False, compute_shap=True, random_labels=True, onesite=onesite_) 
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print("site : ",onesite_)
        print(f"The function took {hours}h {minutes}m {seconds}s to run.") 
```
and
```python
# without random permutations 
from utils import get_predict_sites_list 
for onesite_ in get_predict_sites_list(): # or run each LOSO CV site/fold separately (about 2h computation type by fold)
    classif_vbm_ROI(classif = "svm", datasize = 800, save = False, N763=False, compute_shap=True, random_labels=False, onesite=onesite_) 
```
#### run a statistical analysis of linearly correlated ROI with diagnosis
- call function perform_tests() in univariate_stats.py with VBM=True for VBM ROI and SBM=True for SBM ROI to save a
dataframe in a xlsx file containing a summary of the relationships between age, sex, site, and features using an ordinary least squares and an ANOVA test. The xlsx file is saved in the folder "results_feat_imptce_and_univ_stats".

### SHAP analysis
- create dataframe with shap values and their means (the mean values for each ROI between all test set subjects concatenated for all folds as well as by fold (the mean of test sets subjects for each fold)), save the dataframes as xlsx in "models/ShapValues/shap_computed_from_all_Xtrain/" folder **run make_shap_df() in feature_imptce_shap_analysis.py** with SBM=True for SBM ROI (implemented for the Destrieux atlas, and including subcortical ROI) or with VBM=True (implemented for Neuromorphometrics atlas).
-


