import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler
from collections import namedtuple

DATA_DIR = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
LH_DATA = DATA_DIR + "surface-lh_data.npy"
RH_DATA = DATA_DIR + "surface-rh_data.npy"
META_DATA = DATA_DIR + "metadata.tsv"

SetItem = namedtuple("SetItem", ["train", "test"])

class ClassificationDataset(Dataset):
    """
    Dataset for binary classification with cortical surface data.
    Applies initial transforms (downsampling etc) in __init__ and on-the-fly transforms in __getitem__.
    """
    
    def __init__(self, data_dict, metadata, indices, transforms=None, initial_transform=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping modality names to data arrays.
        metadata : pd.DataFrame
            Metadata dataframe.
        indices : array-like
            Indices to use for this dataset split.
        transforms : dict or None
            Dictionary mapping modality names to on-the-fly transform functions.
        initial_transform : function or None
            Initial transform function (downsampling + axis swapping).
        """

        if initial_transform is not None:
            print("Applying initial transforms (downsampling + axis swapping)...")
            self.data = {}
            for mod, d in data_dict.items():
                print(f"Applying initial transform to {mod}: {d.shape} -> ", end="")
                # Apply initial transform to full data, then subset
                transformed_data = initial_transform(d)  # (n_subjects, n_metrics, n_vertices_downsampled)
                self.data[mod] = transformed_data[indices]  # Subset to split indices
                print(f"{self.data[mod].shape}")
        else:
            # No initial transform, just subset the data
            self.data = {mod: d[indices] for mod, d in data_dict.items()}
        
        self.metadata = metadata.iloc[indices].reset_index(drop=True)
        self.labels = self.metadata["dx"].values.astype(int)
        self.transforms = transforms or {}
        
        print(f"Dataset initialized with {len(self.labels)} samples")
        print(f"Label distribution: {np.bincount(self.labels)}") # index 0 : count of zeros, index 1: count of ones
        # print(f"N_HC/N_BD: {round(np.bincount(self.labels)[0]/np.bincount(self.labels)[1],3)}") # ratio number of healthy controls/ nb of BD subjects in train set

    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index.
        
        Returns
        -------
        tuple
            (transformed_data, label) where transformed_data is a dict of tensors.
        """
        # Get raw data for all modalities
        raw_data = {mod: self.data[mod][idx] for mod in self.data}
        # raw_data values have shape (n_metrics, n_vertices_downsampled) after initial transform
        
        # Apply on-the-fly transforms 
        transformed_data = {}
        for mod, data in raw_data.items():
            # Convert to tensor first 
            data = torch.tensor(data, dtype=torch.float32)
            
            # Apply on-the-fly transform if available
            if mod in self.transforms and self.transforms[mod] is not None:
                try:
                    transformed_data[mod] = self.transforms[mod](data)
                except Exception as e:
                    print(f"Error transforming {mod}: {e}")
                    # Fallback: use data as-is
                    transformed_data[mod] = data
            else:
                # No transform: use data as-is
                transformed_data[mod] = data
        
        label = self.labels[idx]

        return transformed_data, label, idx

    def __len__(self):
        return len(self.labels)


class DataManager:
    """
    Data manager.
    Handles loading and splitting of multimodal cortical data.
    With initial and on-the-fly transforms.
    """
    
    def __init__(self, modalities, split_dict, split_key, data_dir=DATA_DIR, 
                 metadata_file=META_DATA, transforms=None, initial_transform=None):
        """
        Initialize the data manager.
        
        Parameters
        ----------
        modalities : list
            List of modality names (must match .npy file names).
        split_dict : dict
            Dictionary with train/test splits.
        split_key : str
            Key to use for selecting the split.
        data_dir : str
            Directory containing the data files.
        metadata_file : str
            Path to metadata file.
        transforms : dict or None
            Dictionary mapping modality names to on-the-fly transform functions.
        initial_transform : function or None
            Initial transform function.
        """
        self.modalities = modalities
        self.transforms = transforms or {mod: None for mod in modalities}
        self.initial_transform = initial_transform
        self.data_dir = data_dir
        
        print(f"Loading data for modalities: {modalities}")
        
        # Load all data for all modalities
        self.data_dict = {}
        for mod in modalities:
            data_path = os.path.join(data_dir, f"{mod}.npy")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            print(f"Loading {data_path}...")
            all_data = np.load(data_path)  # Shape: (n_subjects, n_vertices, n_metrics)
            self.data_dict[mod] = all_data
            print(f"Loaded {mod}: {all_data.shape}")

        # Load metadata
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        print(f"Loading metadata from {metadata_file}...")
        self.metadata = pd.read_csv(metadata_file, sep="\t")
        print(f"Metadata shape: {self.metadata.shape}")
        
        # Make sure we have dx column for classification
        if "dx" not in self.metadata.columns:
            raise ValueError("Metadata must contain 'dx' column for classification")
        
        # Store train/test split indices
        print("split_key", split_key)
        if split_key not in split_dict:
            raise KeyError(f"Split key '{split_key}' not found in split_dict")
        
        self.train_indices = split_dict[split_key]["train"]
        self.test_indices = split_dict[split_key]["test"]
        
        print(f"Split '{split_key}': {len(self.train_indices)} train, {len(self.test_indices)} test")
        
        # Validate indices
        max_idx = len(self.metadata)
        if np.max(self.train_indices) >= max_idx or np.max(self.test_indices) >= max_idx:
            raise ValueError("Split indices exceed dataset size")

    def get_dataset(self, split):
        """
        Get dataset for specified split.
        
        Parameters
        ----------
        split : str
            'train' or 'test'.
        
        Returns
        -------
        ClassificationDataset
            Dataset for the specified split.
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        indices = self.train_indices if split == "train" else self.test_indices
        print("split : ",split)
        return ClassificationDataset(
            data_dict=self.data_dict,
            metadata=self.metadata,
            indices=indices,
            transforms=self.transforms,
            initial_transform=self.initial_transform
        )


def get_dataloader(split_dict, split_key, modalities, batch_size, transforms=None,
                  initial_transform=None, num_workers=3, pin_memory=True):
    """
    Create train and test dataloaders.
    
    Parameters
    ----------
    split_dict : dict
        Dictionary containing train/test splits.
    split_key : str
        Key for selecting the specific split to use.
    modalities : list
        List of modality names.
    batch_size : int
        Batch size for dataloaders.
    transforms : dict or None
        On-the-fly transform functions for each modality.
    initial_transform : function or None
        Initial transform function (applied once in dataset creation).
    num_workers : int
        Number of dataloader workers.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.
    
    Returns
    -------
    SetItem
        Named tuple with train and test dataloaders.
    """
    print("Creating data manager...")
    manager = DataManager(
        modalities=modalities,
        split_dict=split_dict,
        split_key=split_key,
        transforms=transforms,
        initial_transform=initial_transform
    )

    print("Creating datasets...")
    train_dataset = manager.get_dataset("train")
    test_dataset = manager.get_dataset("test")

    print("Creating dataloaders...")
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Test data should not be shuffled
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Train batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")

    return SetItem(train=trainloader, test=testloader)


def validate_data_compatibility(data_dir=DATA_DIR, modalities=None):
    """
    Validate that data files exist and have compatible shapes.
    
    Parameters
    ----------
    data_dir : str
        Directory containing data files.
    modalities : list or None
        List of modalities to check. If None, checks default modalities.
    
    Returns
    -------
    dict
        Dictionary with validation results.
    """
    if modalities is None:
        modalities = ["surface-lh_data", "surface-rh_data"]
    
    results = {}
    shapes = {}
    
    print("Validating data compatibility...")
    
    # Check metadata
    metadata_path = os.path.join(data_dir, "metadata.tsv")
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path, sep="\t")
        results["metadata"] = {
            "exists": True,
            "shape": metadata.shape,
            "has_dx": "dx" in metadata.columns
        }
        print(f"Metadata: {metadata.shape}, has 'dx': {'dx' in metadata.columns}")
    else:
        results["metadata"] = {"exists": False}
        print("Metadata file not found!")
    
    # Check data files
    for mod in modalities:
        data_path = os.path.join(data_dir, f"{mod}.npy")
        if os.path.exists(data_path):
            try:
                data = np.load(data_path, mmap_mode='r')  # Memory-mapped for large files
                shapes[mod] = data.shape
                results[mod] = {
                    "exists": True,
                    "shape": data.shape,
                    "dtype": str(data.dtype)
                }
                print(f"{mod}: {data.shape}, {data.dtype}")
            except Exception as e:
                results[mod] = {"exists": True, "error": str(e)}
                print(f"{mod}: Error loading - {e}")
        else:
            results[mod] = {"exists": False}
            print(f"{mod}: File not found!")
    
    # Check shape compatibility
    if len(shapes) > 1:
        first_shape = list(shapes.values())[0]
        compatible = all(shape[0] == first_shape[0] for shape in shapes.values())
        results["shape_compatible"] = compatible
        print(f"Shape compatible across modalities: {compatible}")
        
        if results["metadata"]["exists"]:
            n_subjects_data = first_shape[0]
            n_subjects_metadata = results["metadata"]["shape"][0]
            subjects_match = n_subjects_data == n_subjects_metadata
            results["subjects_match"] = subjects_match
            print(f"Number of subjects match (data vs metadata): {subjects_match}")
    
    return results


if __name__ == "__main__":    
    # Validate data
    validation_results = validate_data_compatibility()
    
    if not all(validation_results[mod].get("exists", False) 
              for mod in ["surface-lh_data", "surface-rh_data"]):
        print("Data validation failed. Please check your data files.")
        exit(1)