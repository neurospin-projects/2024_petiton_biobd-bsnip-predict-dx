import numpy as np
import pickle
import torch
import os
import warnings
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from surfify.models import SphericalHemiFusionEncoder
import argparse
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from new_ssl_models import SimCLRFineTunerMinimal
from datasets_biobdbsnip import get_dataloader, DataManager
from data_transforms import create_transforms_for_biobdbsnip

# ---------- CONFIG ----------
MODALITIES = ["surface-lh_data", "surface-rh_data"]
DATA_DIR = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
ROOT = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/2022_cambroise_surfaugment/"
ENCODER_OUTDIR = ROOT + "y_aware_sigma5_save_encoders/" #y_aware_sigma5_save_encoders/" #"test_with_yawareloss_sigma5/" 
PRETRAINED_ENCODER = ENCODER_OUTDIR + "ssl_scnns/checkpoints/1755723336/model_epoch_300_encoder.pth" #1755723336/encoder.pth" #1754614034/encoder.pth" #model_epoch_110_
METRICS = ["thickness", "curv", "sulc"]
SPLITS_PATH = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/dict_splits_test_all_trainset_sizes_N763.pkl" #dict_splits_test.pkl"

# outputs
CHECKPOINT_DIR = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/checkpoints_finetuning_sbm"

def read_pkl(file_path):
    """Read pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(dict_or_array, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_or_array, file)
    print(f'Item saved to {file_path}')

def create_folder_if_not_exists(folder_path):
    """Create folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

def create_filename_from_args(args, suffix=""):
    """Create descriptive filename from model arguments."""
    str_val = "_no_val" if not args.use_validation else ""

    filename = (f"results_{args.mode}_ep{args.epochs}_dr{args.dropout_rate}"
               f"_bs{args.batch_size}_wd{args.weight_decay}"
               f"_pw{args.pos_weight}_lr{args.learning_rate}{suffix}_site{args.site}_gamma{args.gamma}{str_val}")
    return filename

def create_validation_split(train_indices, metadata, val_size=0.2, stratify_col='dx', random_state=42):
    """
    Create validation split from training data.
    
    Parameters
    ----------
    train_indices : array-like
        Original training indices.
    metadata : pd.DataFrame
        Metadata containing labels.
    val_size : float
        Fraction of training data to use for validation.
    stratify_col : str
        Column to stratify on.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    tuple
        (new_train_indices, val_indices)
    """
    # Get labels for stratification
    train_labels = metadata.iloc[train_indices][stratify_col].values
    
    # Split training data into train/validation
    new_train_idx, val_idx = train_test_split(
        range(len(train_indices)),
        test_size=val_size,
        stratify=train_labels,
        random_state=random_state
    )
    
    # Convert back to original indices
    new_train_indices = [train_indices[i] for i in new_train_idx]
    val_indices = [train_indices[i] for i in val_idx]
    
    return new_train_indices, val_indices

def evaluate_model(model, loss_fn, loader, device, desc="Evaluating"):
    """
    Evaluate the model on a dataset.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    loss_fn : callable
        Loss function.
    loader : torch.utils.data.DataLoader
        DataLoader for the dataset.
    device : torch.device
        Device to run evaluation on.
    desc : str
        Description for progress bar.
    
    Returns
    -------
    tuple
        (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    y_pred_list, y_true_list = [], []

    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in pbar:
            try:
                if isinstance(batch, list) and len(batch) == 3:
                    inputs, targets = batch[0], batch[1]
                    if isinstance(inputs, dict):
                        lh_data = inputs.get('surface-lh_data')
                        rh_data = inputs.get('surface-rh_data')
                        if lh_data is not None and rh_data is not None:
                            model_inputs = (lh_data.to(device), rh_data.to(device))
                        else:
                            print("wrong format")
                            quit()
                    else:
                        model_inputs = inputs.to(device)
                        
                    targets = targets.to(device).float()
            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue

            outputs = model(model_inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            y_pred_list.extend(outputs.cpu().numpy())
            y_true_list.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)

    # Calculate metrics
    metrics = {}
    try:
        if len(np.unique(y_true)) >= 2 and not np.all(y_pred == y_pred[0]):
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred > 0)
        else:
            print("there is an issue: len(np.unique(y_true)) < 2 or np.all(y_pred == y_pred[0])")
            quit()
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics["roc_auc"] = 0.0
        metrics["balanced_accuracy"] = 0.0

    return avg_loss, metrics, y_pred, y_true

def train_epoch(model, optimizer, loss_fn, loader, epoch, device, verbose=True):
    """Train the model for one epoch."""
    model.train()
    losses, y_pred, y_true, indices_all = [], [], [], []

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch} training", leave=False)

    for batch_idx, batch in enumerate(loader):
        try:
            if isinstance(batch, list) and len(batch) == 3:
                inputs, targets, indices = batch[0], batch[1], batch[2]
                if isinstance(inputs, dict):
                    lh_data = inputs.get('surface-lh_data')
                    rh_data = inputs.get('surface-rh_data')
                    if lh_data is not None and rh_data is not None:
                        model_inputs = (lh_data.to(device), rh_data.to(device))
                    else:
                        print("Wrong format")
                        quit()
                else:
                    model_inputs = inputs.to(device)
                    
                targets = targets.to(device).float()
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
        
        optimizer.zero_grad()
        outputs = model(model_inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        y_pred.append(outputs.detach().cpu())
        y_true.append(targets.detach().cpu())
        indices_all.append(indices.detach().cpu())

        pbar.update(1)

    pbar.close()

    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    indices_all = torch.cat(indices_all).numpy()

    sort_idx = np.argsort(indices_all)
    y_pred = y_pred[sort_idx]
    y_true = y_true[sort_idx]

    # Calculate training metrics
    metrics = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred > 0)
        
        if verbose:
            print(f"[Epoch {epoch}] Train ROC AUC: {metrics['roc_auc']:.4f}, "
                  f"Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    except Exception as e:
        print(f"Error calculating training metrics: {e}")
        metrics = {"roc_auc": 0.0, "balanced_accuracy": 0.0}

    return model, np.mean(losses), metrics, optimizer, y_pred, y_true

def plot_validation_metrics(val_history, save_path):
    """
    Plot validation metrics over epochs.
    
    Parameters
    ----------
    val_history : dict
        Dictionary containing validation metrics history.
        Expected keys: 'roc_auc', 'balanced_accuracy', 'loss'
    save_path : str
        Path to save the plot.
    """
    epochs = range(1, len(val_history['roc_auc']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Plot ROC AUC
    plt.subplot(1, 3, 1)
    plt.plot(epochs, val_history['roc_auc'], 'b-', label='Validation ROC AUC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('Validation ROC AUC Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)  # ROC AUC is between 0 and 1

    
    # Plot Balanced Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_history['balanced_accuracy'], 'r-', label='Validation Balanced Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title('Validation Balanced Accuracy Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)  # Balanced accuracy is between 0 and 1

    # Plot Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_history['loss'], 'g-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Validation metrics plot saved to {save_path}")


def plot_combined_validation_metrics(val_history, save_path):
    """
    Plot both validation metrics on the same plot.
    
    Parameters
    ----------
    val_history : dict
        Dictionary containing validation metrics history.
    save_path : str
        Path to save the plot.
    """
    epochs = range(1, len(val_history['roc_auc']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_history['roc_auc'], 'b-', label='ROC AUC', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_history['balanced_accuracy'], 'r-', label='Balanced Accuracy', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Validation Metrics Over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 1)
    
    # Add best values as text
    best_auc = max(val_history['roc_auc'])
    best_acc = max(val_history['balanced_accuracy'])
    best_auc_epoch = val_history['roc_auc'].index(best_auc) + 1
    best_acc_epoch = val_history['balanced_accuracy'].index(best_acc) + 1
    
    plt.text(0.02, 0.98, f'Best ROC AUC: {best_auc:.4f} (Epoch {best_auc_epoch})', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.text(0.02, 0.88, f'Best Balanced Acc: {best_acc:.4f} (Epoch {best_acc_epoch})', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined validation metrics plot saved to {save_path}")


def main(args):
    """Main fine-tuning function."""
    args.device = torch.device("cuda" if args.cuda else "cpu")
    n_features = len(METRICS)
    args.conv_filters = [int(item) for item in args.conv_filters.split("-")]
    
    # Create output directory
    create_folder_if_not_exists(args.checkpoint_dir)
    
    # Building encoder backbone with same architecture as SSL training
    encoder = SphericalHemiFusionEncoder(
        n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
        conv_flts=args.conv_filters, activation="ReLU",
        batch_norm=False, conv_mode="DiNe",
    )

    if args.mode == "RIDL":
        print("Initializing encoder with random weights...")
        encoder = encoder.to(args.device)
        print("Encoder initialized with random weights!")
    elif args.mode == "TL":
        print("Loading pre-trained encoder...")
        state_dict = torch.load(args.pretrained_encoder, map_location=args.device, weights_only=True)
        encoder.load_state_dict(state_dict)
        encoder = encoder.to(args.device)
        print("Encoder loaded successfully!")
    else:
        print("Mode has to be RIDL or TL")
        quit()

    # Create fine-tuning model using encoder backbone
    model=SimCLRFineTunerMinimal(encoder, dropout_rate=args.dropout_rate).to(args.device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Setup classification training
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32, device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.gamma, step_size=args.step_size)

    # Load split data
    print("Setting up data transforms...")
    split_data = read_pkl(args.split_dict_path)
    split_key = args.site + "_" + str(args.traindata_size)
    if split_key not in split_data:
        raise KeyError(f"Split key '{split_key}' not found in split data")
    
    original_train_indices = split_data[split_key]["train"]
    test_indices = split_data[split_key]["test"]
    
    # Create validation split if requested
    if args.use_validation:
        print(f"Creating validation split ({args.val_size:.1%} of training data)")
        
        # Load metadata for stratification
        import pandas as pd
        metadata_path = os.path.join(DATA_DIR, "metadata.tsv")
        metadata = pd.read_csv(metadata_path, sep="\t")
        
        # Create train/validation split
        train_indices, val_indices = create_validation_split(
            original_train_indices, 
            metadata, 
            val_size=args.val_size,
            random_state=args.seed
        )
        
        print(f"Original training: {len(original_train_indices)} samples")
        print(f"New training: {len(train_indices)} samples")
        print(f"Validation: {len(val_indices)} samples")
        print(f"Test: {len(test_indices)} samples")
        
    else:
        print("No validation split - using all training data")
        train_indices = original_train_indices
        val_indices = None
    
    print(f"Using {len(train_indices)} training samples for StandardScaler fitting only")
    
    # Create transforms
    #initial_transform, on_the_fly_transforms = --> if using on the fly transforms
    initial_transform, _ = create_transforms_for_biobdbsnip( 
        modalities=MODALITIES,
        data_dir=DATA_DIR,
        ico_order=args.ico_order,
        metrics=METRICS,
        overwrite=args.overwrite_scalers,
        train_indices=train_indices
    )
    print("Transforms ready!")

    # Create data loaders
    print("Loading data...")
    
    # Modify split_data to use new indices
    modified_split_data = {
        split_key: {
            "train": train_indices,
            "test": test_indices
        }
    }
    
    # Create train and test loaders
    loader = get_dataloader(
        split_dict=modified_split_data,
        split_key=split_key,
        modalities=MODALITIES,
        batch_size=args.batch_size,
        transforms= None , #on_the_fly_transforms
        initial_transform=initial_transform
    )
    
    # Create validation loader if needed
    val_loader = None
    if args.use_validation:
        val_split_data = {
            f"{split_key}_val": {
                "train": val_indices,  # Use val_indices as "train" for DataManager
                "test": test_indices   # Dummy test (won't be used)
            }
        }
        
        val_data_manager = DataManager(
            modalities=MODALITIES,
            split_dict=val_split_data,
            split_key=f"{split_key}_val",
            transforms=None, #on_the_fly_transforms,
            initial_transform=initial_transform
        )
        
        val_dataset = val_data_manager.get_dataset("train")  # Get the "train" which is actually validation
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            drop_last=False
        )
        print(f"Validation loader created with {len(val_dataset)} samples")

    print("Data loaders ready!")

    # Training loop
    print("Starting training...")
    val_history = {'roc_auc': [], 'balanced_accuracy': [], 'loss': []}
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        model, mean_loss, train_metrics, optimizer, y_pred_tr, y_true_tr = train_epoch(
            model, optimizer, loss_fn, loader.train, epoch=epoch+1, device=args.device, verbose=True
        )

        # Validation
        if args.use_validation and val_loader is not None:
            val_loss, val_metrics, _, _ = evaluate_model(
                model, loss_fn, val_loader, args.device, desc=f"Validation Epoch {epoch+1}"
            )
            
            # store validation history
            val_history['loss'].append(val_loss)
            val_history['roc_auc'].append(val_metrics['roc_auc'])
            val_history['balanced_accuracy'].append(val_metrics['balanced_accuracy'])
            
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, "
                  f"Val ROC AUC: {val_metrics['roc_auc']:.4f}, "
                  f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
            
            # best validation performance
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_epoch = epoch + 1
                print(f"🎯 New best validation ROC AUC: {best_val_auc:.4f}")

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1} - Train Loss: {mean_loss:.4f}")

    # Save validation plots if validation was used
    if args.use_validation and args.save_plots and len(val_history['roc_auc']) > 0:
        base_filename = create_filename_from_args(args)
        
        # Save separate plots
        plot_path_separate = os.path.join(args.checkpoint_dir, f"{base_filename}_validation_metrics_separate.png")
        plot_validation_metrics(val_history, plot_path_separate)
        
        # # Save combined plot
        # plot_path_combined = os.path.join(args.checkpoint_dir, f"{base_filename}_validation_metrics.png")
        # plot_combined_validation_metrics(val_history, plot_path_combined)

    # Testing
    if loader.test is not None:
        print(f"\nEvaluating on LOSO test set (subjects from left-out site {args.site}) ...")
        
        test_loss, test_metrics, test_y_pred, test_y_true = evaluate_model(
            model, loss_fn, loader.test, args.device, desc="Testing"
        )

        print(f'\nFinal Test Metrics:')
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

        # save results
        results = {
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'validation_history': val_history if args.use_validation else None,
            'best_val_auc': best_val_auc if args.use_validation else None,
            'best_epoch': best_epoch if args.use_validation else None,
            'args': vars(args)
        }

        if not args.use_validation:
            results['train_predictions'] = {'y_pred': y_pred_tr, 'y_true': y_true_tr}
            results['test_predictions'] = {'y_pred': test_y_pred, 'y_true': test_y_true}
            print(f"Saved ordered predictions - Train: {len(y_pred_tr)}, Test: {len(test_y_pred)}")
                
        base_filename = create_filename_from_args(args)
        
        from pathlib import Path
        encoder_name = Path(args.pretrained_encoder).name

        results_path_txt = os.path.join(args.checkpoint_dir, f"{encoder_name}{base_filename}_results.txt")
        results_path_pkl = os.path.join(args.checkpoint_dir, f"{encoder_name}{base_filename}_results.pkl")
        save_pkl(results, results_path_pkl)
        
        with open(results_path_txt, "w") as f:
            f.write("=== MODEL CONFIGURATION ===\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Dropout rate: {args.dropout_rate}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Weight decay: {args.weight_decay}\n")
            f.write(f"Pos weight: {args.pos_weight}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Site: {args.site}\n")
            f.write(f"Gamma: {args.gamma}\n")
            f.write(f"Training data size: {args.traindata_size}\n")
            if args.use_validation:
                f.write(f"Validation size: {args.val_size}\n")
                f.write(f"Best validation ROC AUC: {best_val_auc:.4f} (Epoch {best_epoch})\n")
            if args.mode=="TL":
                f.write(f"pretrained encoder : {args.pretrained_encoder}")
            f.write("\n=== RESULTS ===\n")
            f.write(pprint.pformat(results))
        print(f"Results saved to {results_path_txt}")

    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SSL pre-trained encoder for binary classification")
    # dict containing ratio of training set nbs of HC subjects / BD subjects for each LOSO CV site
    pos_weight_dict = {"Baltimore":1.175,
                       "Boston": 1.26,
                       "Dallas": 1.188,
                       "Detroit": 1.19,
                       "Hartford": 1.18,
                       "mannheim": 1.268,
                       "creteil": 1.244,
                       "udine": 1.082,
                       "galway": 1.286,
                       "pittsburgh": 1.379,
                       "grenoble": 1.292,
                       "geneve": 1.24
                        }
    assert len(pos_weight_dict.keys())==12

    # Encoder parameters (default parameters matche SSL training)
    parser.add_argument("--mode", default="TL", type=str, help="training mode", choices=["RIDL","TL"])
    parser.add_argument("--ico-order", default=5, type=int, help="Icosahedron order")
    parser.add_argument("--latent-dim", default=128, type=int, help="Latent dimension")
    parser.add_argument("--fusion-level", default=1, type=int, help="Fusion level")
    parser.add_argument("--conv-filters", default="128-128-256-256", type=str, help="Conv filters")
    
    # Data parameters
    parser.add_argument("--site", type=str, default="Baltimore", help="Site name")
    parser.add_argument("--traindata_size", type=int, default=8, choices=list(range(9)),
                       help="Training data size")
    parser.add_argument("--split_dict_path", type=str, default=SPLITS_PATH, help="Path to splits")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--pos_weight", type=float, default=1.584, help="Positive class weight")
    
    # Validation parameters
    parser.add_argument("--use_validation", action="store_true", 
                       help="Use validation set for hyperparameter tuning")
    parser.add_argument("--val_size", type=float, default=0.1, # 0.05 used for validation of hyperparams except for val of pretraining encoders (0.1 then) 
                       help="Fraction of training data to use for validation")
    parser.add_argument("--save_plots", action="store_true", 
                       help="Save validation metric plots")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for validation split")
    
    # Scheduler parameters
    parser.add_argument("--gamma", type=float, default=0.85, help="Learning rate decay factor")
    parser.add_argument("--step_size", type=int, default=10, help="Learning rate decay step size")
    
    # Saving parameters
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint frequency")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, 
                       help="Checkpoint directory")
    
    # Data processing
    parser.add_argument("--overwrite_scalers", action="store_true", 
                       help="Overwrite existing scalers")
    
    # System parameters
    parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA")
    
    # Model parameters
    parser.add_argument("--dropout_rate", type=float, default=0.2, 
                       help="Dropout rate for classifier")
    parser.add_argument("--pretrained-encoder", type=str, default=PRETRAINED_ENCODER, 
                       help="Pretrained encoder pth file")

    args = parser.parse_args()
    args.pos_weight = pos_weight_dict[args.site]
    print("traindata_size ", args.traindata_size)

    if not torch.cuda.is_available():
        args.cuda = False
        print("CUDA is not available and has been disabled.")

    main(args)

