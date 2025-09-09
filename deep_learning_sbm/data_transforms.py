import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from surfify.utils import icosahedron, downsample_data, downsample
import joblib
import os
from .augmentations import Normalize, Reshape, Transformer

ROOT = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/2022_cambroise_surfaugment/"

"""
Adapted from https://github.com/neurospin-projects/2022_cambroise_surfaugment/tree/master 
C. Ambroise, V. Frouin, B. Dufumier, E. Duchesnay and A. Grigis, 
MixUp brain-cortical augmentations in self-supervised learning, Machine Learning in Clinical Neuroimaging (MLCN) 2023.
"""

def create_initial_transforms(ico_order=5):
    """
    Create initial transforms (downsampling + axis swapping)
    """
    order = 7
    ico_verts, _ = icosahedron(order)
    down_indices = []
    for low_order in range(order - 1, ico_order - 1, -1):
        low_ico_verts, _ = icosahedron(low_order)
        down_indices.append(downsample(ico_verts, low_ico_verts))
        ico_verts = low_ico_verts
    
    def transform_func(x):
        """Exact match with evaluate_representations.py"""
        downsampled_data = downsample_data(x, 7 - ico_order, down_indices)
        return np.swapaxes(downsampled_data, 1, 2)
    
    return transform_func

def files_are_equal(path1, path2):
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        return f1.read() == f2.read()
    
def create_train_only_scaler(modalities, data_dir, train_indices, initial_transform, 
                           ico_order=5, metrics=None, overwrite=False):
    """
    Create and fit StandardScaler on training data only.
    
    Parameters
    ----------
    modalities : list
        List of modality names.
    data_dir : str
        Directory containing the data.
    train_indices : array-like
        Training sample indices.
    initial_transform : function
        Initial transform function.
    ico_order : int
        Icosahedron order.
    metrics : list
        List of cortical metrics.
    overwrite : bool
        Whether to overwrite existing scalers.
    
    Returns
    -------
    dict
        Dictionary mapping modality names to fitted StandardScaler objects.
    """
    metrics = metrics if metrics is not None else ['thickness', 'curv', 'sulc']
    ico_verts, _ = icosahedron(ico_order)
    input_shape = (len(metrics), len(ico_verts))
    
    scalers = {}
    
    for mod in modalities:

        # to make sure that the initial transforms applied are the same: 
        # I generated the transforms using the datamanager and transforms from corentin's code on biobd bsnip and compare them to those here 
        # they are the same (prints True)
        # scaler_path_from_corentin_code = f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/2022_cambroise_surfaugment/experiments/biobdbsnip/{mod}_scaler.save"
        # print("scaler path",scaler_path_from_corentin_code)
        # print(files_are_equal(scaler_path_from_corentin_code, os.path.join(data_dir, f"{mod}_scaler.save")))

        scaler_path = os.path.join(data_dir, f"{mod}_scaler.save")
        
        # Check if scaler already exists
        if not overwrite and os.path.exists(scaler_path):
            print(f"Loading existing scaler for {mod}")
            scaler = joblib.load(scaler_path)
            scalers[mod] = scaler
            continue
        
        # Load data and apply initial transform
        data_path = os.path.join(data_dir, f"{mod}.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading data and fitting scaler for {mod} on training data only...")
        original_data = np.load(data_path)  # Shape: (n_subjects, n_vertices, n_metrics)
        
        # Apply initial transform
        transformed_data = initial_transform(original_data)  # Shape: (n_subjects, n_metrics, n_vertices)
        
        # only use training data for fitting the scaler
        train_data = transformed_data[train_indices]  # Shape: (n_train, n_metrics, n_vertices)
        
        # Flatten training data for scaler fitting
        train_data_flat = train_data.reshape(len(train_indices), -1)  # Shape: (n_train, n_metrics*n_vertices)
        
        # Fit StandardScaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_data_flat)
        
        # Save scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        
        scalers[mod] = scaler
        print(f"StandardScaler fitted on {len(train_indices)} training samples for {mod}")
        print(f"Scaler saved to {scaler_path}")
    
    return scalers


def create_transforms_for_biobdbsnip(modalities, data_dir, ico_order=5, 
                                    metrics=None, train_indices=None, overwrite=False, on_the_fly=False):
    """
    Create transforms
    Only the StandardScaler is fitted on training data - everything else uses the current dataset.
    
    Parameters
    ----------
    modalities : list
        List of modality names.
    data_dir : str
        Directory containing the data.
    ico_order : int
        Icosahedron order.
    metrics : list
        List of cortical metrics.
    train_indices : array-like or None
        Training indices for fitting StandardScaler. If None, fits on all data.
    overwrite : bool
        Whether to overwrite existing scalers.
    
    Returns
    -------
    tuple
        (initial_transform, on_the_fly_transforms_dict)
    """
    metrics = metrics if metrics is not None else ['thickness', 'curv', 'sulc']
    ico_verts, _ = icosahedron(ico_order)
    input_shape = (len(metrics), len(ico_verts))
    
    print("Creating transforms for modalities:", modalities)
    print("Using metrics:", metrics)
    
    initial_transform = create_initial_transforms(ico_order=ico_order)
    
    if train_indices is not None:
        print("Fitting StandardScalers on training data")
        scalers = create_train_only_scaler(
            modalities=modalities,
            data_dir=data_dir,
            train_indices=train_indices,
            initial_transform=initial_transform,
            ico_order=ico_order,
            metrics=metrics,
            overwrite=overwrite
        )
    else:
        print("train_indices not provided, StandardScaler cannot be fitted on all data")
        quit()
    
    # Create on-the-fly transform pipelines using the fitted scalers
    on_the_fly_transforms = {}
    
    for mod in modalities:
        scaler = scalers[mod]
        
        # Create the exact pipeline from evaluate_representations.py
        # transforms.Compose([
        #     Reshape((1, -1)),          # flatten
        #     scaler.transform,          # standardize  
        #     transforms.ToTensor(),     # convert to tensor
        #     torch.squeeze,             # remove batch dimension
        #     Reshape(input_shape),      # reshape to final shape
        # ])
        # + Normalize()
        
        def create_transform_pipeline(fitted_scaler, input_shape, on_the_fly):
            def transform_func(x):
                """                
                parameters -
                x : torch.Tensor
                    Shape: (features, vertices) e.g., (3, 10242)
                
                returns -
                torch.Tensor
                    Transformed tensor, shape: (features, vertices)
                """
                if on_the_fly:
                    # Reshape((1, -1)) - flatten
                    x_flat = x.reshape(1, -1)  # (1, features*vertices)
                    
                    # scaler.transform - standardize (SCALER IS FITTED ON TRAINING DATA ONLY)
                    x_numpy = x_flat.cpu().numpy() if x_flat.is_cuda else x_flat.detach().numpy()
                    x_scaled = fitted_scaler.transform(x_numpy)  # (1, features*vertices)
                    
                    # transforms.ToTensor() - convert to tensor
                    x_tensor = torch.from_numpy(x_scaled).float()
                    
                    # torch.squeeze - remove batch dimension
                    x_squeezed = torch.squeeze(x_tensor)  # (features*vertices,)
                    
                    # Reshape(input_shape) - reshape to final shape
                    x_reshaped = x_squeezed.reshape(input_shape)  # (features, vertices)
                    
                    # Normalize() - normalize
                    normalize = Normalize()
                    x_normalized = normalize(x_reshaped) 

                    return x_normalized

                # else: 
                #     normalize = Normalize()
                #     x_normalized = normalize(x)
                #return x_normalized
                
                return x
            
            return transform_func
        
        on_the_fly_transforms[mod] = create_transform_pipeline(scaler, input_shape, on_the_fly)
    
    print("Transform pipelines ready!")
    return initial_transform, on_the_fly_transforms


def create_ssl_compatible_transforms(modalities, data_dir, ico_order=5, 
                                   metrics=None, fit_scalers=True, overwrite=False,
                                   train_indices=None):
    if not fit_scalers:
        print("Warning: fit_scalers=False, scalers may not be fitted properly")
    
    return create_transforms_for_biobdbsnip(
        modalities=modalities,
        data_dir=data_dir,
        ico_order=ico_order,
        metrics=metrics,
        train_indices=train_indices,
        overwrite=overwrite
    )