import pickle
import pandas as pd


def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(dict_or_array, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_or_array, file)
    print(f'Item saved to {file_path}')

DATA_DIR = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
META_DATA = DATA_DIR+"metadata.tsv"
metadata = pd.read_csv(META_DATA, sep="\t")

print(metadata)

sizes = [75, 150, 200, 300, 400, 450, 500, 600, 700]
results={}

for idx, size in enumerate(sizes):
    print("idx ",idx," size ",size)
    path = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_classif/classifSBM/L2LR_N"+str(size)+"_Destrieux_SBM_ROI_N763.pkl"
    dat = read_pkl(path)

    for site in ["Baltimore", "Boston", "Dallas", "Detroit", "Hartford", "mannheim", "creteil", 
                "udine", "galway", "pittsburgh", "grenoble", "geneve"]:
        # Make a lookup dictionary: participant_id → index
        id_to_index = {pid: idx for idx, pid in metadata["participant_id"].items()}

        # Get indices preserving order of each list
        indices_list1 = [id_to_index[pid] for pid in dat[site]["participant_ids_tr"] if pid in id_to_index]
        indices_list2 = [id_to_index[pid] for pid in dat[site]["participant_ids_te"] if pid in id_to_index]
        results[site+"_"+str(idx)]={"train":indices_list1, "test":indices_list2}

print(results.keys())
# save_pkl(results,"dict_splits_test_all_trainset_sizes_N763.pkl")

