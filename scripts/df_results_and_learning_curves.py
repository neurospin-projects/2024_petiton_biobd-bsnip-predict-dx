import  numpy as np, os
from utils import read_pkl, get_predict_sites_list, save_pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp, ttest_ind
from numpy import trapz

"""
1. SAVES DATAFRAME WITH ALL CLASSIFICATION RESULTS FOR ML AND DL WITH save_all_results_dataframe
2. PLOT THE LEARNING CURVES WITH create_plot_learning_curves


"""

PATH_TL_RESULTS="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifVoxelWise/TL/"
PATH_RIDL_RESULTS="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifVoxelWise/RIDL/"
PATH_ML_RESULTS_VOXELWISE="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifVoxelWise/ML/"

PATH_ML_RESULTS_ROIWISE_VBM="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifVBM/"
PATH_ML_RESULTS_ROIWISE_SBM="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifSBM/"
PATH_ML_RESULTS_ROIWISE_SBM_NOSUB="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifSBM/no_subcortical/"
PATH_ML_RESULTS_ROIWISE_SBM_7SUB="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifSBM/with7subcorticalROI/"
PATH_ALL_RESULTS="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/"


def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

def check_dict_values(dictionary):
    # Get the length of the first list value
    first_length = len(next(iter(dictionary.values())))
    
    # Check if all values are lists of the same length
    assert all(len(value) == first_length for value in dictionary.values()), "All values should be lists of the same size"

def get_densenet_dataframe(transfer=False):
    """
        return the dataframe for RIDL and TL ROC-AUC and balanced_accuracy measures
        transfer : (bool) whether we want the df with metrics for the TL models or RIDL models
    """
    trainingsizes861 = [100,175,250,350,400,500,600,700,800]
    alldata = {"train_size_category":[],"test_roc_auc": [],"test_balanced_accuracy":[],\
                "fold-site": [], "classifier": [], "feature_type":[], "atlas":[], "dataset":[]}

    for size in trainingsizes861:
        
        print("densenet with training size :",size)
        if transfer : path = PATH_TL_RESULTS+ "densenet"+str(size)
        else : path = PATH_RIDL_RESULTS+ "densenet"+str(size)
        
        for site in get_predict_sites_list():
            alldata["train_size_category"].append(size)
            alldata["fold-site"].append(site)
            filetest = "Test_"+site+"_densenet121_vbm_bipolar_epoch199.pkl"
            datatest = read_pkl(os.path.join(path,filetest))

            alldata["test_roc_auc"].append(datatest["metrics"]['roc_auc on test set'])
            alldata["test_balanced_accuracy"].append(datatest["metrics"]['balanced_accuracy on test set'])
            alldata["atlas"].append("NA")
            alldata["dataset"].append("N861")
            # print(site ,"auc ", round(100*auc,3))

    if transfer: alldata["classifier"] =  ["densenet with transfer"]*len(alldata["test_roc_auc"])
    else : alldata["classifier"] =  ["densenet"]*len(alldata["test_roc_auc"])
    alldata["feature_type"] = ["vbm_voxelwise"]*len(alldata["test_roc_auc"])

    # check that all lists of the dictionnary are of the same length
    # 12*9 : 12 sites, 9 training set sizes
    check_dict_values(alldata)

    df = pd.DataFrame(alldata)
    print(df)
    return df

def save_feature(trainingsizes_, root_, alldata_, feature_type ,atlas=None, dataset=False):
    """
    Aim : fills up alldata_ dataframe with the performance metrics for all training set sizes and all feature types 
            between ROI and voxels, VBM and Freesurfer, and the 763 subjects of 861 subjects from the dataset
    root_ : (str) root where the data is
    alldata_: (pandas df) dataframe being filled up as we call this function
    feature_type : (str) type of feature; can be "vbm_voxelwise", "VBM_ROI", "SBM_ROI","SBM_ROI_no_subcortical", or "SBM_ROI_7subROI"
    atlas : (str) atlas for ROI features, "NA" otherwise
    dataset : (str) dataset used for classification ("N763" or "N861", the number corresponds to the number of subjects in each dataset)
                N763 is a subset of N861
    """
    assert atlas in ["Desikan", "Destrieux","Neuromorphometrics","NA"]
    assert feature_type in ["vbm_voxelwise", "VBM_ROI", "SBM_ROI","SBM_ROI_no_subcortical","SBM_ROI_7subROI"]
    
    print(" feature_type ",feature_type, " atlas ", atlas)
    atlas_str = ""
    if feature_type == "vbm_voxelwise" : 
        feat_str = "_voxelwise"    
    else : 
        feat_str = "_"+feature_type
        atlas_str = "_"+atlas

    if dataset is None: datasetstr =""
    else : datasetstr="_"+str(dataset)

    for size in trainingsizes_:
        for classif in ["EN", "svm", "L2LR" ,"MLP","xgboost"]:
            if feature_type == "vbm_voxelwise" :  filenameROI = classif+"_N"+str(size)+atlas_str+feat_str+".pkl"
            else: filenameROI = classif+"_N"+str(size)+atlas_str+feat_str+datasetstr+".pkl"
            data = read_pkl(root_+filenameROI)
            # keys ['Baltimore', 'Boston', 'Dallas', 'Detroit', 'Hartford', 'mannheim', 'creteil', 'udine', 'galway', 'pittsburgh', 'grenoble', 'geneve', 'mean over all sites']
            # keys for data["Baltimore"] for ex: ['roc_auc', 'balanced-accuracy']

            for site in get_predict_sites_list():
                alldata_["train_size_category"].append(size)
                alldata_["classifier"].append(classif)
                alldata_["test_roc_auc"].append(data[site]['roc_auc'])
                alldata_["test_balanced_accuracy"].append(data[site]['balanced-accuracy'])
                alldata_["fold-site"].append(site)
                alldata_["feature_type"].append(feature_type)
                alldata_["atlas"].append(atlas)
                alldata_["dataset"].append(dataset)

def save_all_results_dataframe(save=False):
    trainingsizes763 = [75, 150, 200, 300, 400, 450, 500, 600, 700]
    trainingsizes861 = [100, 175, 250, 350, 400, 500, 600, 700, 800]
    alldata = dict()

    # get deep learning models' performances for each training size 
    df_densenet = get_densenet_dataframe(False) #RIDL
    df_densenetTL = get_densenet_dataframe(True) #TL

    alldata = {"train_size_category":[],"test_roc_auc": [],"test_balanced_accuracy":[],\
                "fold-site": [], "classifier": [], "feature_type":[], "atlas":[], "dataset":[]}

    # for voxel-wise, whole-brain sMRI, vbm features
    save_feature(trainingsizes861, PATH_ML_RESULTS_VOXELWISE, alldata, feature_type = "vbm_voxelwise",  atlas="NA", dataset="N861")
    
    # for ROI vbm features and 763 subjects
    save_feature(trainingsizes763, PATH_ML_RESULTS_ROIWISE_VBM, alldata, feature_type="VBM_ROI", atlas="Neuromorphometrics", dataset="N763")
    
    # same with 861 subjects
    save_feature(trainingsizes861, PATH_ML_RESULTS_ROIWISE_VBM, alldata, feature_type="VBM_ROI", atlas="Neuromorphometrics", dataset="N861")

    # for ROI SBM features without any subcortical measures (only CT and SA measures)
    save_feature(trainingsizes763, PATH_ML_RESULTS_ROIWISE_SBM_NOSUB, alldata,  feature_type="SBM_ROI_no_subcortical", atlas="Destrieux", dataset="N763")
    save_feature(trainingsizes763, PATH_ML_RESULTS_ROIWISE_SBM_NOSUB, alldata,  feature_type="SBM_ROI_no_subcortical", atlas="Desikan", dataset="N763")

    # for SBM ROI features with only the same 7 subcortical ROIs as those used in Nunes et al (and the same CT and SA ROI as available in the atlas)
    save_feature(trainingsizes763, PATH_ML_RESULTS_ROIWISE_SBM_7SUB, alldata,  feature_type="SBM_ROI_7subROI", atlas="Destrieux", dataset="N763")
    save_feature(trainingsizes763, PATH_ML_RESULTS_ROIWISE_SBM_7SUB, alldata,  feature_type="SBM_ROI_7subROI", atlas="Desikan", dataset="N763")

    for atlas_ in ["Desikan", "Destrieux"]:
        # for ROI SBM features (CT, SA, and 17 subcortical measures (per hemisphere) extracted using Freesurfer)
        save_feature(trainingsizes763, PATH_ML_RESULTS_ROIWISE_SBM, alldata, feature_type="SBM_ROI", atlas=atlas_, dataset="N763")

    check_dict_values(alldata)
    df = pd.DataFrame(alldata)
    print(df)
    
    concatenated_df = pd.concat([df, df_densenet, df_densenetTL], ignore_index=True)

    print(concatenated_df)
    if save: 
        filepath =PATH_ALL_RESULTS+'all_classification_results_dataframe.pkl'    
        save_pkl(concatenated_df, filepath)
        print("dataframe saved to : ",filepath)

def get_list_mean_and_std(df, classifier, feature_type, atlas, metric="roc_auc", dataset="N763", verbose=False):
    """
    Aim : returns the lists of mean ROC-AUC values and corresponding standard deviations over all LOSO-CV sites
            for each training set size (the lists' lengths are equal tot he number of training set sizes, which is equal to 9),
            as well as the AUC (area under curve) for the corresponding learning curve
            these three measures are for one classifier at a time only.
    df : (pandas DataFrame) the dataframe containing all the classifier's pefromanced for all feature types, datasets, atlases, etc.
    feature_type : (str) the feature type used for classification (VBM voxels, VBM ROI, SBM ROI, SBM ROIs with varying numbers of additional subcortical ROIs)
    atlas : (str) the atlas for ROI; for voxelwise, atlas = "NA"
    metric : (str) either 'roc_auc' or 'balanced_accuracy', the metric whose mean values over LOSO-CV folds we want the function to return
    dataset : (str) either 'N763' or 'N861' the dataset used for classification. SBM ROI can only have N763 as only 763 participants have SBM ROI measures.
                N763 is a subset of N861
    """

    assert feature_type in ["vbm_voxelwise", "VBM_ROI", "SBM_ROI","SBM_ROI_no_subcortical","SBM_ROI_7subROI", "zscores_VBM_ROI","zscores_SBM_ROI"]
    assert classifier in ["EN", "svm","L2LR" ,"MLP", "xgboost", "densenet", "densenet with transfer"]
    assert dataset in ["N763", "N861"]

    trainingsizes763 = [75, 150, 200, 300, 400, 450, 500, 600, 700]
    trainingsizes861 = [100, 175, 250, 350, 400, 500, 600, 700, 800]
    if dataset =="N763": N_values = trainingsizes763
    if dataset =="N861": N_values = trainingsizes861

    _df = df[(df['classifier'] == classifier) & (df['feature_type'] == feature_type) & (df['atlas'] == atlas) & (df['dataset'] == dataset)]

    # Group by "train_size_category" and calculate mean roc-auc for each size
    mean_roc_auc_by_size = _df.groupby('train_size_category')["test_"+metric].mean()
    mean_roc_auc_list = mean_roc_auc_by_size.tolist()
    if verbose : print(mean_roc_auc_list)

    std_roc_auc_by_size = _df.groupby('train_size_category')["test_"+metric].std() # change to sem for standard error instead of std, or .std() for standard error
    std_roc_auc_list = std_roc_auc_by_size.tolist()

    actual_area = trapz(mean_roc_auc_list, N_values)
    ideal_area = np.trapz(np.ones_like(N_values), N_values)  # where y ==1
    auc = (actual_area / ideal_area) * 100

    return mean_roc_auc_list, std_roc_auc_list, auc

def plot_line(means, N_values, color_dict, classifier, label_dict, diffauc, cpt, linestyle_="solid", alpha=1):
    """
        plot line for one classifier for learning curve plot
    """
    if cpt!=0 and cpt!=10 : 
        sns.lineplot(x=N_values, y=means, linewidth=5,\
                               label=label_dict[classifier]+" ($\Delta$ = "+str(round(diffauc,2))+")", \
                                color= color_dict[classifier],linestyle=linestyle_, alpha=alpha) 
    elif cpt ==10:
        sns.lineplot(x=N_values, y=means, linewidth=5, label=label_dict[classifier]+" (AUC = "+str(round(diffauc,2))+")", \
                     color= color_dict[classifier],linestyle=linestyle_, alpha=alpha)

    else : 
        sns.lineplot(x=N_values, y=means, linewidth=5,  color= color_dict[classifier],linestyle=linestyle_,alpha=alpha)
    
    

def create_plot_learning_curves(feature="ROI", metric="roc_auc", SBM_atlas="Destrieux", dataset="N763"): 
    """
    feature (str): possible values: 
        - "ROI" creates a plot comparing learning curves where SBM (freesurfer) ROI and VBM (cat12) ROI are used as features
            for the 5 ML classifiers (L2LR, EN, svm, MLP, gradient boosting)
        - "ROI_VBM_vs_Nunes" compares VBM ROI with Desikan SBM ROI with 7 (per hemisphere, 14 total) additional Freesurfer-extracted
            subcortical ROIs (replicated Nunes)
        - "voxels" creates the learning curves to compare all classifiers trained with voxel-wise images as features
            (the same 5 classifiers as before + DL classifiers, which are one RI-DL model and one TL model)
        - "ROI_no_subcortical" plots the learning curves of VBM ROI vs SBM ROI without any subcortical ROI (for SBM of course)
        - "SBM_ROI_subcortical" plots the learning curves of classification with all (CT, Area) freesurfer-derived measures as features 
            as well as the learning curves of classifiers using the same measures alongside 17 subcortical ROIs (for the selected SBM_atlas).
        - "SBM_ROI_atlases" compares the Desikan and Destrieux atlases, both with either 17 subcortical features

    metric (str) : "roc_auc" or "balanced_accuracy" : the metric values to plot
    SBM_atlas (str): 'Destrieux' or 'Desikan' : the atlas for SBM ROI measures
    dataset (str) : "N763" or "N861": the dataset (the number is the size of the dataset) 
                    used for classification / "N861" will produce an error if the comparison is made between SBM ROI features
                    and any other features as the maximum number of subjects available with SBM ROI measures is 763

    """
    assert metric in ["roc_auc","balanced_accuracy"], "wrong classification performance metric, it must be either 'roc_auc' or 'balanced_accuracy'"
    assert feature in ["ROI", "voxels", "SBM_ROI_subcortical","SBM_ROI_atlases","ROI_no_subcortical","ROI_VBM_vs_Nunes"], "wrong feature type"
    assert dataset in ["N861", "N763"]," wrong dataset name ! it must be eitehr 'N861' or 'N763' !"
    assert SBM_atlas in ["Desikan", "Destrieux"], "wrong atlas"
    
    # load results' dataframe
    df = read_pkl(PATH_ALL_RESULTS+'all_classification_results_dataframe.pkl' )
    print(df["feature_type"].unique())
    print(df["classifier"].unique())

    # print(df)
    # to print some results at maximum training set size, uncomment below :
    """
    vbm_df = df[(df['train_size_category'] == 700) & (df['feature_type'] == "VBM_ROI") & (df['atlas'] == "Neuromorphometrics") & (df['dataset'] == "N763")]
    vbm_df861 = df[(df['train_size_category'] == 800) & (df['feature_type'] == "VBM_ROI") & (df['atlas'] == "Neuromorphometrics") & (df['dataset'] == "N861")]

    sbm_df = df[(df['train_size_category'] == 700) & (df['feature_type'] == "SBM_ROI") & (df['atlas'] == SBM_atlas) & (df['dataset'] == "N763")]
    sbm_df_Nunes_replication = df[(df['train_size_category'] == 700) & (df['feature_type'] == "SBM_ROI_7subROI") & (df['atlas'] == "Desikan") & (df['dataset'] == "N763")]

    print("VBM roc auc at maximum training set size (700) for each ML classifier")
    print(round(100*vbm_df.groupby("classifier")["test_roc_auc"].mean(),2))

    print("VBM roc auc at maximum training set size (800) for each ML classifier")
    print(round(100*vbm_df861.groupby("classifier")["test_roc_auc"].mean(),2))

    print("\nSBM roc auc at maximum training set size (700) for each ML classifier")
    print(round(100*sbm_df.groupby("classifier")["test_roc_auc"].mean(),2))

    print("\n sbm_df_Nunes_replication (Desikan SBM with 7 subcortical ROIs per hemisphere)")
    print(np.round(100*sbm_df_Nunes_replication.groupby("classifier")["test_roc_auc"].mean(),2))
    """

    # compares VBM ROI vs SBM ROI with "SBM_atlas" as SBM ROI atlas and Neuromorphometrics as VBM ROI atlas for dataset N763
    # and for all subcortical ROIs
    if feature =="ROI" : 
        list_network_names = ["L2LR","EN", "svm" ,"MLP","xgboost"] 
        features = ["SBM_ROI","VBM_ROI"]
        assert dataset=="N763", "you cannot compare VBM ROI and SBM ROI as features for classification with more subjects used for VBM compared to SBM"
        atlases = [SBM_atlas, "Neuromorphometrics"]

    if feature =="ROI_VBM_vs_Nunes" : # comparison VBM ROI with all subcortical measures vs SBM with 7 subcortical measures and Desikan atlas (replicated Nunes)
        list_network_names = ["L2LR","EN", "svm" ,"MLP","xgboost"] 
        features = ["SBM_ROI_7subROI","VBM_ROI"]
        assert dataset=="N763", "you cannot compare VBM ROI and SBM ROI as features for classification with more subjects used for VBM compared to SBM"
        atlases = ["Desikan", "Neuromorphometrics"]

    if feature =="ROI_no_subcortical" : 
        list_network_names = ["L2LR","EN", "svm" ,"MLP","xgboost"] 
        features = ["SBM_ROI_no_subcortical", "VBM_ROI"] 
        atlases = [SBM_atlas, "Neuromorphometrics"]

    if feature == "voxels" : 
        list_network_names = ["L2LR","EN", "svm" ,"MLP","xgboost" ,"densenet", "densenet with transfer"]
        features = ["vbm_voxelwise"]
        atlases = ["NA"]
        assert dataset == "N861","for voxel-wise classification the dataset has to be set to N861"

    if feature =="SBM_ROI_subcortical" : 
        list_network_names = ["L2LR","EN", "svm", "L2LR" ,"MLP","xgboost"] 
        features = ["SBM_ROI", "SBM_ROI_no_subcortical"]
        atlases = [SBM_atlas]*2

    if feature == "SBM_ROI_atlases":
        list_network_names= ["L2LR","EN", "svm" ,"MLP","xgboost"] 
        # change to "SBM_ROI_no_subcortical" or "SBM_ROI_7subROI" for comparison of SBM ROI without subcortical measures or with the 7 (per hemisphere) subcortical measures used in Nunes et al
        features = ["SBM_ROI"] 
        atlases = ["Desikan", "Destrieux"]

    print(features)

    # list of training set sizes
    # N_values = list(np.linspace(100, 900, 9, dtype=int)) 
    trainingsizes763 = [75, 150, 200, 300, 400, 450, 500, 600, 700]
    trainingsizes861 = [100, 175, 250, 350, 400, 500, 600, 700, 800]
    if dataset =="N763": N_values = trainingsizes763
    if dataset =="N861": N_values = trainingsizes861

    print(N_values)

    fig, ax1 = plt.subplots()
    deep_palette = sns.color_palette("deep")

    color_dict={
        "EN" : deep_palette[1],
        "xgboost" : deep_palette[2],
        "L2LR" : deep_palette[3],
        "svm" : deep_palette[4],
        "MLP":"#8B4513",
        "densenet":"#FFC0CB",
        "densenet with transfer":deep_palette[0]
    }

    label_dict={
        "EN" : "EN",
        "xgboost" : "Gradient Boosting",
        "L2LR" : "L2LR",
        "svm" : "SVM-RBF",
        "MLP":"MLP",
        "densenet":"RI-DL",
        "densenet with transfer":"TL"
    }

    dict_area_under_curve = {"L2LR":{},"EN":{}, "svm":{} ,"MLP":{},"xgboost":{},"densenet":{},"densenet with transfer":{}}
    
    
    for classifier_ in list_network_names:
        cpt = 0
        if len(features)==2: diffauc = 0
        assert len(features)<=2
        for plot_feature in features: 
            if "VBM_ROI" in plot_feature: assert atlases[cpt]=="Neuromorphometrics"
            
            means, stds, auc = get_list_mean_and_std(df, classifier_, plot_feature, atlases[cpt], metric, dataset=dataset)
            dict_area_under_curve[classifier_][plot_feature]=auc
            # print(plot_feature, " area under the curve ", auc, " classifier ",classifier_)
            if cpt==0 or plot_feature=="vbm_voxelwise": diffauc = auc
            else: 
                diffauc = diffauc-auc
                # print("difference in auc : ", round(diffauc,2))

            if plot_feature=="SBM_ROI" or plot_feature=="SBM_ROI_7subROI":
                plot_line(means, N_values, color_dict, classifier_,label_dict, diffauc, cpt, linestyle_="dashed")
            elif plot_feature=="VBM_ROI":
                plot_line(means, N_values, color_dict, classifier_, label_dict, diffauc, cpt, linestyle_="solid")
            elif plot_feature=="vbm_voxelwise": 
                if classifier_ in ["densenet", "densenet with transfer"]: plot_line(means, N_values, color_dict, classifier_, label_dict, diffauc, 10, linestyle_="dashed")
                else : plot_line(means, N_values, color_dict, classifier_, label_dict, diffauc, 10, linestyle_="solid")
            # print(classifier_,"  ",plot_feature, " mean ",round(100*means[len(means)-1],3), "std ",round(100*stds[len(stds)-1],3))
            means = np.array(means)
            stds= np.array(stds)
            ax1 = plt.gca()
            cpt+=1
               
    ax1.set_xlabel("training dataset size", fontsize=25)
    if metric=="roc_auc": ax1.set_ylabel("Mean ROC-AUC over LOSO test sites", fontsize=25)
    if metric=="balanced_accuracy": ax1.set_ylabel("Mean Balanced-Accuracy over LOSO test sites", fontsize=25)

    legend1 = ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=20)
    
    if feature =="ROI" or "ROI_VBM_vs_Nunes":
        # with or without SBM atlas name
        ax1.set_title("VBM ROI vs SBM ROI classification performance", fontsize=30) 
        # ax1.set_title("VBM ROI vs "+SBM_atlas+" SBM ROI classification performance", fontsize=30)
    elif feature == "SBM_ROI_subcortical" : 
        ax1.set_title(SBM_atlas+" SBM ROI classification with no subcortical measures, with 7 subcortical measures, and with 17 subcortical ROIs", fontsize=30)
    elif feature == "ROI_no_subcortical":
        ax1.set_title("VBM ROI vs "+SBM_atlas+" SBM ROI without subcortical measures", fontsize=30)
    elif feature == "SBM_ROI_atlases":
        ax1.set_title("SBM ROI with Desikan atlas vs Destrieux atlas", fontsize=30)
    else: 
        ax1.set_title("VBM voxel-wise classification performance", fontsize=30)

    if feature !="voxel": 
        line = Line2D([0], [0], color='grey', linestyle='dashed')
        line.set_linewidth(4) # choose width of grey line in legend
        line_gray = Line2D([0], [0], color='grey', label='Regular Gray Line')
        line_gray.set_linewidth(4)

    if feature=="ROI" or feature == "ROI_VBM_vs_Nunes":
        str_VBM = ", AUC : L2LR ("+str(round(dict_area_under_curve['L2LR']["VBM_ROI"],2))+"), EN ("+str(round(dict_area_under_curve['EN']["VBM_ROI"],2))+\
                "), svm ("+str(round(dict_area_under_curve['svm']["VBM_ROI"],2))+"), MLP ("+str(round(dict_area_under_curve['MLP']["VBM_ROI"],2))+\
                    "), GB ("+ str(round(dict_area_under_curve['xgboost']["VBM_ROI"],2))+")"
        print("\n",str_VBM)
        print("mean VBM auc across all classifiers :" , round(np.mean([v["VBM_ROI"] for v in dict_area_under_curve.values() if "VBM_ROI" in v]),1),"\n")

        if feature == "ROI_VBM_vs_Nunes": sbmROI = "SBM_ROI_7subROI"
        else : sbmROI = "SBM_ROI"
        str_SBM = ", AUC : L2LR ("+str(round(dict_area_under_curve['L2LR'][sbmROI],2))+"), EN ("+str(round(dict_area_under_curve['EN'][sbmROI],2))+\
                    "), svm ("+str(round(dict_area_under_curve['svm'][sbmROI],2))+"), MLP ("+str(round(dict_area_under_curve['MLP'][sbmROI],2))+\
                        "), GB ("+ str(round(dict_area_under_curve['xgboost'][sbmROI],2))+")"
        print("\n",str_SBM)
        print("mean SBM ",sbmROI," auc across all classifiers :" , round(np.mean([v[sbmROI] for v in dict_area_under_curve.values() if sbmROI in v]),1),"\n")
        legend2 = ax1.legend([line_gray, line],  ['VBM'+str_VBM,'SBM'+str_SBM],loc="upper left", bbox_to_anchor=(0, 0.11), fontsize=20)

    ax1.add_artist(legend1)
    ax1.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
   
def get_ttest_result(df_metrics_one, df_metrics_two, metric): 
    """
    Aim : perform t-test on the difference of the values of df slices df_metrics_one minus the values of df_metrics_two
            H0 : difference is equal to zero
            with metric ("roc_auc" or "balanced_accuracy") measures

    df_metrics_one : (pandas DataFrame slice) df slice one
    df_metrics_two : (pandas DataFrame slice) df slice two
    metric: (str) "roc_auc" or "balanced_accuracy"
    """
    result = df_metrics_one.values - df_metrics_two.values

    print("first mean ", metric, " ", round(100*np.mean(df_metrics_one.values),2)," std ", round(100*np.std(df_metrics_one.values),2), \
        "\nsecond mean ",metric, " ", round(100*np.mean(df_metrics_two.values),2), " std ", round(100*np.std(df_metrics_two.values),2))
    print("first - second mean "+metric+" values over LOSO folds: \n",np.round(result,2))
    print("\nfirst - second mean "+metric+" values over LOSO folds: \n",np.round(100*np.mean(result),2))

    t_statistic, p_value = ttest_1samp(list(result), 0)

    # Print the results
    print("T-statistic:", round(t_statistic,3))
    print("P-value:", p_value)

def ttest_ROI_vbmVSfreesurfer(metric="roc_auc",features_to_compare="SBM_atlases", SBM_atlas="Destrieux"):
    assert features_to_compare in ["ROI","ROIvs7sub", "SBM_atlases", "SBM_atlases_no_subcortical", \
                                   "SBM_Destrieux_nosubROI_17subROI", "SBM_Desikan_nosubROI_17subROI"]
    df = read_pkl(PATH_ALL_RESULTS+'all_classification_results_dataframe.pkl' )

    vbm_df = df[(df['train_size_category'] == 700) & (df['feature_type'] == "VBM_ROI") & (df['atlas'] == "Neuromorphometrics") & (df['dataset'] == "N763")]
    vbm_df861 = df[(df['train_size_category'] == 800) & (df['feature_type'] == "VBM_ROI") & (df['atlas'] == "Neuromorphometrics") & (df['dataset'] == "N861")]

    sbm_df = df[(df['train_size_category'] == 700) & (df['feature_type'] == "SBM_ROI") & (df['atlas'] == SBM_atlas) & (df['dataset'] == "N763")]
    sbm_df_basic = df[(df['train_size_category'] == 700) & (df['dataset'] == "N763")]
    sbm_df_Nunes_replication = df[(df['train_size_category'] == 700) & (df['feature_type'] == "SBM_ROI_7subROI") & (df['atlas'] == "Desikan") & (df['dataset'] == "N763")]
    
    for classifier in ["L2LR","EN", "svm" ,"MLP","xgboost"] :
        print(classifier)
        if features_to_compare=="ROI":
            get_ttest_result(vbm_df[vbm_df["classifier"]==classifier]["test_"+metric], sbm_df[sbm_df["classifier"]==classifier]["test_"+metric], metric)
        if features_to_compare=="ROIvs7sub":
            get_ttest_result(vbm_df[vbm_df["classifier"]==classifier]["test_"+metric], sbm_df_Nunes_replication[sbm_df_Nunes_replication["classifier"]==classifier]["test_"+metric], metric)
        if features_to_compare=="SBM_atlases":
            get_ttest_result(sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI") & (df['atlas'] == "Destrieux")]["test_"+metric], \
                sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI") & (df['atlas'] == "Desikan")]["test_"+metric], metric)
        if features_to_compare=="SBM_atlases_no_subcortical":
            get_ttest_result(sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI_no_subcortical") & (df['atlas'] == "Destrieux")]["test_"+metric], \
                sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI_no_subcortical") & (df['atlas'] == "Desikan")]["test_"+metric], metric)
        if features_to_compare=="SBM_Destrieux_nosubROI_17subROI":
            get_ttest_result(sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI_no_subcortical") & (df['atlas'] == "Destrieux")]["test_"+metric], \
                sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI") & (df['atlas'] == "Destrieux")]["test_"+metric], metric)
        if features_to_compare=="SBM_Desikan_nosubROI_17subROI":
            get_ttest_result(sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI_no_subcortical") & (df['atlas'] == "Desikan")]["test_"+metric], \
                sbm_df_basic[(sbm_df_basic["classifier"]==classifier) & (df['feature_type'] == "SBM_ROI") & (df['atlas'] == "Desikan")]["test_"+metric], metric)




def main():
    # save the dataframe containing the roc auc and balanced accuracy 
    # metrics for all classification experiments
    # save_all_results_dataframe(save=True)
    # create_plot_learning_curves(feature="ROI", metric="roc_auc", SBM_atlas="Destrieux",dataset="N763")
    ttest_ROI_vbmVSfreesurfer(metric="roc_auc",features_to_compare="ROI", SBM_atlas="Destrieux")
    quit()

    # create_plot_learning_curves(feature="ROI", metric="roc_auc", SBM_atlas="Desikan",dataset="N763")

    # create_plot_learning_curves(feature="ROI_VBM_vs_Nunes", metric="roc_auc",SBM_atlas="Desikan",dataset="N763")

    # t-tests to get the results of Results paragraph one
    # comparing VBM ROI to SBM ROI (Destrieux + 17 subcortical ROI)
    # ttest_ROI_vbmVSfreesurfer(metric="roc_auc", features_to_compare="ROI", SBM_atlas="Destrieux")

    # comparing VBM ROI to SBM ROI (Destrieux + 7 subcortical ROI)
    # ttest_ROI_vbmVSfreesurfer(metric="roc_auc", features_to_compare="ROIvs7sub", SBM_atlas="Destrieux")

    # comparing SBM ROI (Destrieux + 7 subcortical ROI) to SBM ROI (Desikan + 7 subcortical ROI)
    # ttest_ROI_vbmVSfreesurfer(metric="roc_auc",features_to_compare="SBM_atlases" )
    # ttest_ROI_vbmVSfreesurfer(metric="roc_auc",features_to_compare="SBM_atlases_no_subcortical" )
    # ttest_ROI_vbmVSfreesurfer(metric="roc_auc",features_to_compare="SBM_Destrieux_nosubROI_17subROI" )
    # ttest_ROI_vbmVSfreesurfer(metric="roc_auc",features_to_compare="SBM_Desikan_nosubROI_17subROI" )

    # voxel-wise VBM plot
    create_plot_learning_curves(feature="voxels", metric="roc_auc", SBM_atlas="Destrieux",dataset="N861")






if __name__ == "__main__":
    main()

