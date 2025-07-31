from utils import get_predict_sites_list, read_pkl
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from itertools import combinations
import pickle, os

ROOT = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
RESULTSFOLDER=ROOT+"results_classif/classifVoxelWise/"
TLPATH=RESULTSFOLDER+"TL/deep_ensemble/"
RIDLPATH=RESULTSFOLDER+"RIDL/deep_ensemble/"

def save_pkl(dict_or_array, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_or_array, file)
    print(f'Item saved to {file_path}')

def get_mean(list_models= [1,2,3,4,5,6,7,8,9,10], transfer=False, metric = "roc_auc", \
             verbose=False, test = True, train=False, model ="densenet"):
    """
    Parameters : 
        transfer (bool) : whether to get the 10-ensemble values for transfer learning models or randomly initialized models.
        metric (string) : either 'roc_auc' or 'balanced_accuracy' : what metric to print when verbose = True
        verbose (bool) : whether to call the print() when running the function
    Aim :
        get the mean of the predictions on the testing set over 10 transfer or randomnly-initialized densenets.
    """
    assert not (test and train), "you must choose either train or test predictions"
    assert not (model=="alexnet" and transfer), "alexnet has not been trained for TL task"
    assert metric in ["roc_auc" , "balanced_accuracy"], "Metric must be 'roc_auc' or 'balanced_accuracy'."
    
    all_ypred_by_densenet = {
            "site": [],
            "y_pred": [],
            "y_true": [],
            "densenet_nb": [],
            metric: []
        }
    
    for nb in list_models:
        # if verbose: print(model," nb ",nb)
        meanauc = []
        
        for site in get_predict_sites_list():
            if transfer: path = TLPATH+ model+str(nb)
            else : path = RIDLPATH+ model+str(nb)
            
            if model=="densenet":
                if test: filename = "/Test_"+site+"_densenet121_vbm_bipolar_epoch199.pkl"
                if train : filename = "/Train_"+site+"_densenet121_vbm_bipolar_epoch199.pkl"
            if model == "alexnet": 
                if test : filename = "/Test_"+site+"_alexnet_vbm_bipolar_epoch199.pkl"
                if train: 
                    print("training set metrics not saved for alexnet")
                    quit()

            if test : data = read_pkl(path+filename)
            if train : data = read_pkl(path+filename) 
            metric_ = data["metrics"][metric+" on test set"]

            ypred = data["y_pred"]
            ytrue = data["y_true"]

            # auc for one site, one model :
            # if verbose : print(site ,' roc auc : ',metric_)
            meanauc.append(metric_)

            all_ypred_by_densenet["y_pred"].append(np.array(ypred))
            all_ypred_by_densenet["y_true"].append(np.array(ytrue))
            all_ypred_by_densenet["site"].append(site)
            all_ypred_by_densenet["densenet_nb"].append(nb)
            all_ypred_by_densenet[metric].append(metric_)

    # print(all_ypred_by_densenet)
    dataframe_all_results = pd.DataFrame(all_ypred_by_densenet)
    if verbose : print(dataframe_all_results)

    
    # mean over sites
    mean_metric_by_site = dataframe_all_results.groupby(['site'])[metric].mean()
    mean_metric_by_site = mean_metric_by_site.reset_index()

    # mean over models
    mean_metric_by_densenet = dataframe_all_results.groupby(['densenet_nb'])[metric].mean()
    mean_metric_by_densenet = mean_metric_by_densenet.reset_index()

    if verbose : 
        print(mean_metric_by_densenet)
        print("before DE: mean over all models and sites respectively : ",round(100*np.mean(list(mean_metric_by_densenet[metric])),2),\
                       " = ",round(100*np.mean(list(mean_metric_by_site[metric])),2))
        print("before DE: std over all models and sites respectively: ",round(100*np.std(list(mean_metric_by_densenet[metric])),2),\
                       " != ",round(100*np.std(list(mean_metric_by_site[metric])),2))
    
    all_equal = True
    # # Group by "site" and iterate over each group
    for site, group in dataframe_all_results.groupby('site'):
        # Extract the "y_true" values for current site
        y_true_values = group['y_true'].values
        
        # Check if all arrays in the current site are equal
        if not all((x == y_true_values[0]).all() for x in y_true_values):
            all_equal = False
            break
    assert all_equal, "The true y labels are not all the same for each site"

    # compute mean y_pred for each site
    mean_ypred_by_site = dataframe_all_results.groupby(['site'])['y_pred'].mean()
    if verbose : print(mean_ypred_by_site)

    # get y_true array for each site
    y_true_for_each_site = dataframe_all_results.groupby('site')['y_true'].first()
    # if verbose : print(y_true_for_each_site)

    # Initialize a dictionary to store the ROC AUC (or balanced accuracy) values for each site
    metric_values = {}

    for index in y_true_for_each_site.index:
        if metric == "roc_auc": metric_ = roc_auc_score(y_true_for_each_site[index], mean_ypred_by_site[index])
        else : 
            ypred_acc = (mean_ypred_by_site[index] > 0)
            metric_ = balanced_accuracy_score(y_true_for_each_site[index], ypred_acc)
        metric_values[index] = metric_

    metric_series = pd.Series(metric_values)

    if verbose : 
        print(round(metric_series,3))
        if metric == "roc_auc": print("ensemble mean ROC-AUC over models :",round(100*np.mean(metric_series),2))
        else: print("ensemble mean Balanced Accuracy over models :",round(100*np.mean(metric_series),2))

    return mean_ypred_by_site.to_dict(),  y_true_for_each_site.to_dict()


def print_deep_ensemble_results(model_="densenet", transfer_=False):
    """
        Aim : Returns the mean ROC-AUC, mean Balanced Accuracy, and standard deviations between 
                LOSO-CV sites and 5-DE models (there are 252 5-DE models in this case).
        model_ : (str), "densenet" or "alexnet", type of model 
        transfer_ : (bool) if True, get the results of TL models, otherwise get the results for RIDL models 
    """
    print("transfer : ",transfer_)
    assert model_ in ["densenet","alexnet"]
    assert not(transfer_ and model_=="alexnet"), "alexnet not trained in TL setting"
    str_tl= "TL" if transfer_ else "RIDL"

    data = read_pkl(RESULTSFOLDER+""+str_tl+"/5DE_"+str_tl+"_all_combinations_ypred_"+model_+".pkl")
    df = pd.DataFrame.from_dict(data, orient='index')
    _, dict_ytrue = get_mean([1,2,3,4,5], transfer=transfer_, metric = "roc_auc", verbose=False, test = True, train=False, model =model_)

    def compute_roc_auc(row):
        return {col: roc_auc_score(dict_ytrue[col], row[col]) for col in df.columns}
    
    def compute_balanced_accuracy(row):
        return {col: balanced_accuracy_score(dict_ytrue[col], (row[col]>0)) for col in df.columns}

    # Apply function to each row and convert to DataFrame
    roc_auc_df = df.apply(compute_roc_auc, axis=1).apply(pd.Series)
    print(roc_auc_df)

    print(round(100*np.mean(roc_auc_df.mean(axis=0)),2),"% ROC AUC , +/- ", \
          round(100*np.std(roc_auc_df.mean(axis=0)),2), "%") # mean and std btw models (axis = 0)
    print(round(100*np.mean(roc_auc_df.mean(axis=1)),2),"% ROC AUC , +/-  ", \
          round(100*np.std(roc_auc_df.mean(axis=1)),2), "%") # mean and std btw sites (axis = 1)
    
    balanced_accuracy_df = df.apply(compute_balanced_accuracy, axis=1).apply(pd.Series)
    print(round(100*np.mean(balanced_accuracy_df.mean(axis=0)),2)," % balanced accuracy , +/-  ",\
           round(100*np.std(balanced_accuracy_df.mean(axis=0)),2),"%") # mean and std btw models (axis = 0)
    print(round(100*np.mean(balanced_accuracy_df.mean(axis=1)),2)," % balanced accuracy , +/- ", \
          round(100*np.std(balanced_accuracy_df.mean(axis=1)),2),"%") # mean and std btw sites (axis = 1)


def save_dict_5DE_all_combinations(model_="densenet", transfer_=False, test_=True, train_=False):
    """
        Aim : save dictionary with scores for all 252 5-DE combinations
        model_ : (str), "densenet" or "alexnet", type of model 
        transfer_ : (bool) if True, get the results of TL models, otherwise get the results for RIDL models 
    """
    assert not(transfer_ and model_=="alexnet"), "alexnet not trained in TL setting"
    assert not (train_ and test_), "train or test split must be chosen"
    dict_all_combinations_DE = {}
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    combinations_of_five = list(combinations(numbers, 5))  
    nb_combination = 0
    for combination in combinations_of_five :
        list_models_voxel_wise = list(combination)
        nb_combination+=1
        dict_mean_ypred, dict_ytrue = get_mean(list_models_voxel_wise, transfer=transfer_, metric = "roc_auc", verbose=False, test = test_, train = train_, model =model_)
        dict_all_combinations_DE[nb_combination]=dict_mean_ypred
    print(f"Total number of combinations: {len(combinations_of_five)}")
    str_tl = "TL" if transfer_ else "RIDL"
    str_tr_te = "_test" if test_ else "_train"
    save_pkl(dict_all_combinations_DE, RESULTSFOLDER+str_tl+"/5DE_"+str_tl+"_all_combinations_ypred_"+model_+str_tr_te+".pkl")
   
def main():
    # print alexnet RIDL without 5-DE results over all 10 trained models :
    # get_mean([1,2,3,4,5,6,7,8,9,10], transfer=False, metric = "roc_auc", verbose=True, test = True, train=False, model ="alexnet")
    # save_dict_5DE_all_combinations(model_="alexnet") # save the 252 combinations making up 5-DE RIDL models with alexnet 

    # print_deep_ensemble_results(model_="densenet", transfer_=True) # print alexnet 5-DE RIDL results
    save_dict_5DE_all_combinations(model_="densenet", transfer_=True, test_=False, train_=True)

    # to get the overall mean metrics over the 10 RIDL and 10 TL models
    # get_mean([1,2,3,4,5,6,7,8,9,10], transfer=True, metric = "roc_auc", verbose=True, test = True, train=False, model ="densenet")

    # to save the dictionaries with all classification scores for all test set subjects for all 252 combinations for RIDL and TL models
    # save_dict_5DE_all_combinations(True)
    # save_dict_5DE_all_combinations(False)

if __name__ == "__main__":
    main()

