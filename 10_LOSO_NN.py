
import os, torch, pickle, subprocess
import numpy as np
from collections import OrderedDict

# torch imports
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms.transforms import Compose

# model backbones
from densenet import densenet121
from alexnet import AlexNet3D_Dropout

# data transforms
from transforms import Crop, Padding, Normalize

# dataset
from BD_dataset import BipolarDataset

# model performance evaluation
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# miscellaneous
from tqdm import tqdm
from collections import namedtuple
from copy import deepcopy
import re
import argparse

# "homemade" utils
from utils import get_predict_sites_list, create_folder_if_not_exists


#inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
PRETRAINING_MODEL="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/TL/DenseNet_HCP_IXI_window-0.25_0_epoch_30.pth"

#outputs
OUTPUT_DIR=ROOT+"testfolder/"

SetItem = namedtuple("SetItem", ["train", "test"])
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels", "indices"])


def get_state_dict(path):
    """
    Parameters : 
        path : 
            model path (we used models saved with '.pth' format)
    Aim :
        Return the state dictionary of model saved at path 'path'.
    Outputs : 
        dict :
            state dictionary of the model.
    """

    assert path.endswith(".pth"),"wrong file type. expecting pth file."

    if torch.cuda.is_available() and torch.cuda.device_count()>0: 
        state_dict = torch.load(path)
    else : 
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    # specific to the models we saved, if uncessesary return directly state_dict : 
    dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict["model"].items())
    return dict

def get_optimizer(path):
    """
    Parameters : 
        path : 
            model path (we used models saved with '.pth' format)
    Aim :
        Return the optimizer of model saved at path 'path'.
    Outputs : 
        dict :
            optimizer of the model.
    """

    saved_model = torch.load(path, map_location=lambda storage, loc: storage)
    model_optim = saved_model["optimizer"]
    return model_optim


def save_metrics(outdir, epoch, metrics_dict, site, split="Train", exp_name=""):    
    """Save metrics dictionary to a .pkl file."""
    assert split in ["Train","Test"]
    filename = f"{split}_{exp_name}_epoch_{epoch}_site_{site}.pkl" if split=="Train" \
               else f"{split}_{site}_{exp_name}_epoch{epoch}.pkl"
    
    filepath = os.path.join(outdir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(metrics_dict, f)
    
    print(f"{split} metrics saved to: {filepath}")

def checkpoint(site, model, epoch, outdir, name=None, optimizer=None, scheduler=None):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: Net
        the network model.
    epoch: int
        the epoch index.
    fold: int
        the fold index.
    outdir: str
        the destination directory where a 'model_<fold>_epoch_<epoch>.pth'
        file will be generated.
    optimizer: Optimizer, default None
        the network optimizer (save the hyperparameters, etc.).
    scheduler: Scheduler, default None
        the network scheduler.
    kwargs: dict
        others parameters to save.
    """

    outfile = os.path.join(outdir, name+"_epoch_"+str(epoch)+"_"+site+".pth")
    
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None
        }, outfile)
    
    return outfile

def collate_fn(list_samples):
    """ 
    Adapted from https://github.com/Duplums/SMLvsDL.
    Parameters : 
        list_samples : 
            list of samples using indices from sampler.
    Aim :
        the function passed as the collate_fn argument is used to collate lists
            of samples into batches. A custom collate_fn is used here to apply the transformations.
    Outputs : 
        DataItem :
        named tuple named 'DataItem' containing 3 items named "inputs", "outputs", and "labels".            
    See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
    """
    data = dict(outputs=None) 
    data["inputs"] = torch.stack([torch.from_numpy(np.array(sample[0])) for sample in list_samples], dim=0).float()
    data["labels"] = torch.stack([torch.tensor(np.array(sample[1],dtype=float)) for sample in list_samples], dim=0).squeeze().float()
    data["indices"] = torch.tensor([sample[2] for sample in list_samples], dtype=torch.long)
    return DataItem(**data)

def get_dataloader(site_, train=False, test=False, **args):
    """ 
    Adapted and simplified from https://github.com/Duplums/SMLvsDL. 
    Parameters : 
        **args : dictionary of parameters such as 'pb' (scz or bipolar), 'root' (rootpath), and 'batch_size'.
        train : not required. Change train to = True and uncomment last "if" statement if you want to retrieve the training set.
    Aim :
        Returns the DataLoader objects containing the data for training and testing for tiher the bipolar dataset or schizophrenia dataset.
    Outputs : 
        SetItem(train=trainloader, test=testloader)
        named tuple called "SetItem" containing two DataLoader objects named "test" and "train"
    """

    print("test : ", test, "train : ", train)
    print("Loading the bipolar dataset ...")
    dataset = dict()

    # these transforms are important !
    input_transforms = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),  Normalize()]) if args["transforms"] else None        
    
    testloader, trainloader = None, None

    if train:
        dataset["train"] = BipolarDataset(site=site_, preproc="vbm", split="train", transforms=input_transforms, datasize_idx=args["datasize_idx"])
        trainloader = DataLoader(
            dataset["train"], batch_size=args['batch_size'], sampler=RandomSampler(dataset["train"]),
            collate_fn=collate_fn, num_workers= 3, pin_memory=True, drop_last=False)        
        print('...train dataset :', type(dataset['train']), np.shape(dataset['train']),"...")

    if test :
        dataset["test"] = BipolarDataset(site=site_,preproc="vbm", split="test",transforms=input_transforms, datasize_idx=args["datasize_idx"])   
        testloader = DataLoader(
            dataset["test"], batch_size= args['batch_size'], 
            collate_fn = collate_fn, num_workers= 3, pin_memory=True, drop_last=False)
        
        print('...test dataset :', type(dataset['test']), np.shape(dataset['test']),"...")

    return SetItem(train=trainloader, test=testloader)

def train(model, optimizer, loss_fn, loader, epoch=None, device=None, verbose=True):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    loss_fn : callable
        Loss function.
    loader : torch.utils.data.DataLoader
        Training data loader.
    epoch : int, optional
        Current epoch number (for logging).
    device : str
        Device to run training on.
    verbose : bool
        Whether to print batch-level debug info.

    Returns
    -------
    model : torch.nn.Module
    mean_loss : float
    metrics : dict
    optimizer : torch.optim.Optimizer
    """
    if device is not None: model.to(device)
    model.train()

    pbar = tqdm(total=len(loader), desc=f"Mini-bacth epoch {epoch} training", leave=False)
    # leave=False to clear the progress bar once it reaches 100% to keep only final output (keeps logs clean)

    losses, y_pred, y_true, indices_all = [], [], [], []

    for batch_idx, batch in enumerate(loader):
        # print(f"Batch {batch_idx + 1} - inputs shape: {batch.inputs.shape}") # shape of input data
        inputs = batch.inputs.to(device, non_blocking=True)

        targets = [t.to(device) for t in (batch.outputs, batch.labels) if t is not None]
        targets = targets[0] if len(targets) == 1 else targets
        if targets.ndim == 0:  # scalar → add batch dimension
            targets = targets.unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = loss_fn(outputs, targets)
        batch_loss.backward()
        optimizer.step()

        losses.append(batch_loss.item())
        y_pred.extend(outputs.detach().cpu().numpy())
        y_true.extend(targets.detach().cpu().numpy())
        indices_all.extend(batch.indices.numpy())

        pbar.update()

    pbar.close()

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert len(y_pred.shape) == 1, "The vector of predictions y does not have the right number of dimensions"
    indices_all = np.array(indices_all)

    # reordering y_pred and y_true by indices_all to the original dataset order
    # (in order to save the predictions in the right order despite the random sampler)
    sort_idx = np.argsort(indices_all)
    y_pred = y_pred[sort_idx]
    y_true = y_true[sort_idx]

    metrics = {
        "y_pred": y_pred,
        "y_true": y_true,
        "roc_auc on train set": roc_auc_score(y_true, y_pred),
        "balanced_accuracy on train set": balanced_accuracy_score(y_true, y_pred > 0)
    }

    if verbose:
        print(f"[Epoch {epoch}] ROC AUC: {metrics['roc_auc on train set']:.4f}, "
              f"Balanced Acc: {metrics['balanced_accuracy on train set']:.4f}")

    return model, np.mean(losses), metrics, optimizer

def test(model, loss_fn, loader: DataLoader, **args):
    """
    Adapted and simplified from https://github.com/Duplums/SMLvsDL. 

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate (e.g., densenet121).
    loss_fn : callable
        Loss function used for computing test loss (here: nn.BCEWithLogitsLoss).
    loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    **args : dict
        Additional parameters such as:
        - 'device': device to run inference ('cpu' or 'cuda').
        - 'checkpoint_dir': directory to save outputs.
        - 'exp_name': experiment name for saved files.

    Returns
    -------
    y : np.ndarray
        Predicted outputs.
    X : np.ndarray
        Input data.
    y_true : np.ndarray
        Ground truth labels.
    loss : float
        Average loss over the test set.
    metrics : dict
        Dictionary of evaluation metrics (ROC AUC, balanced accuracy).
    """
    model.to(args["device"])
    model.eval()
    nb_batches = len(loader)
    pbar = tqdm(total=nb_batches, desc="Testing Mini-Batches", leave=False)
    total_loss = 0.0

    y, y_true, X = [], [], []
    metrics = {}

    if args["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device specified but GPU not available.")

    with torch.no_grad():
        for dataitem in loader:
            pbar.update()

            inputs = dataitem.inputs
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(args["device"])

            # Collect targets from possible attributes outputs and labels
            targets_list = []
            for item in (dataitem.outputs, dataitem.labels):
                if item is not None:
                    # If scalar, unsqueeze to add batch dimension
                    if item.numel() == 1:
                        item = item.unsqueeze(0)
                    targets_list.append(item.to(args["device"]))
                    y_true.extend(item.cpu().numpy())

            outputs = model(inputs)

            if targets_list:
                targets_tensor = torch.cat(targets_list, dim=0)  # or torch.stack() depending on shape

                batch_loss = loss_fn(outputs, targets_tensor)
                total_loss += batch_loss.item() / nb_batches

            y.extend(outputs.cpu().numpy())

            if isinstance(inputs, torch.Tensor):
                X.extend(inputs.cpu().numpy())

    pbar.close()

    y_pred = np.array(y)
    y_true = np.array(y_true)
    X = np.array(X)

    metrics["roc_auc on test set"] = roc_auc_score(y_true, y_pred)
    metrics["balanced_accuracy on test set"] = balanced_accuracy_score(y_true, y_pred > 0)

    print(f"ROC AUC on test set: {metrics['roc_auc on test set']:.4f}")
    print(f"Balanced accuracy on test set: {metrics['balanced_accuracy on test set']:.4f}")

    # Ensure checkpoint directory exists
    saving_dir = args.get('checkpoint_dir')
    if saving_dir and not os.path.isdir(saving_dir):
        os.makedirs(saving_dir, exist_ok=True) 
        print(f"Directory {saving_dir} created.")

    return y_pred, X, y_true, total_loss, metrics


def initialize(**args):
    if args["net"]=="densenet": model = densenet121()
    if args["net"]=="alexnet": model = AlexNet3D_Dropout()
    assert model is not None,"no network type chosen"
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.584, dtype=torch.float32, device= args["device"])) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma= 0.8, step_size= 10)
    
    checkpointdir = args['checkpoint_dir']

    return model, loss, optimizer, scheduler, checkpointdir

def load_model_from_path(model_to_test_path, **args):
    if args["net"]=="densenet": model = densenet121()
    if args["net"]=="alexnet": model = AlexNet3D_Dropout()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.584, dtype=torch.float32, device= args["device"])) 
    state_dict = get_state_dict(model_to_test_path)

    try :
        print("Loading the model ...")
        model.load_state_dict(state_dict)
    except BaseException as e:
        print('Error while loading the weights: %s' % str(e))
    print("... state dictionary loaded.")
    return model,loss

def _load_model_weights(model, checkpoint_model, transfer_path):
    """Helper function to load model weights with different checkpoint formats."""
    if hasattr(checkpoint_model, "state_dict"):
        model.load_state_dict(checkpoint_model.state_dict())
    elif isinstance(checkpoint_model, dict) and "model" in checkpoint_model:
        state_dict_transfer = get_state_dict(transfer_path)
        unexpected = model.load_state_dict(state_dict_transfer, strict=False)
        print('Model loading info:', unexpected)
        print('Model loaded')
    else:
        model.load_state_dict(checkpoint_model)

def run(site, nb_epochs_per_saving=10, model_to_test_path = None, **args):
    """
    Run training or testing for a model at a given site.

    Parameters
    ----------
    site : str
        Site name.
    nb_epochs_per_saving : int
        Number of epochs between checkpoints.
    model_to_test_path : str, optional
        Path to a pre-trained model to load for testing only.
    **args : dict
        Additional arguments including:
        - pb : str ("scz" or "bipolar")
        - root : str (root path)
        - nb_epochs : int
        - exp_name : str
        - device : torch device
        - transfer_path : str, optional (path to pretrained weights)
        - checkpoint_dir : str, optional

    Returns
    -------
    results_test : dict
        Dictionary containing predictions, labels, loss, and metrics.
    """
  
    print(f"Site = {site}")

    model, loss_fn, optimizer, scheduler, checkpoint_dir = initialize(**args)
    nb_epochs = args["nb_epochs"]

    # Prevent conflicting modes
    assert not (model_to_test_path and args.get("transfer_path")), \
        "Cannot apply transfer learning when only testing (no training) a model that has already been trained."

    # -----------------------------
    # Training mode
    # -----------------------------
    if model_to_test_path is None:  # if not None : case in which we load a model just to test it
        loader = get_dataloader(site, train=True, test=False, **args)
        if loader.train is not None:
            print("... training dataloader loaded.")

        # Transfer learning
        transfer_path = args.get("transfer_path")
        if transfer_path:
            print("Loading checkpoint model ...")
            try:
                checkpoint_model = torch.load(transfer_path, map_location=args["device"])
            except Exception as e:
                raise RuntimeError(f"Cannot load checkpoint: {e}")

            print(f"Performing transfer learning from {transfer_path}")
            _load_model_weights(model, checkpoint_model, transfer_path)

        model = model.to(args["device"])

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print("Multi GPUs available: using DataParallel.")
            model = nn.DataParallel(model)


        # Reinitialize optimizer/scheduler states
        optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        model.load_state_dict(deepcopy(model.state_dict()))
        if scheduler:
            scheduler.load_state_dict(deepcopy(scheduler.state_dict()))

        # Training loop
        for epoch in range(nb_epochs):
            print(f"Epoch {epoch+1}/{nb_epochs}")
            model, loss_val, train_metrics, optimizer = train(
                model, optimizer, loss_fn, loader.train, epoch, device=args["device"]
            )

            if scheduler:
                scheduler.step()
                # print(f'Scheduler lr: {scheduler.get_last_lr()}')
                # print(f'Optimizer lr: {optimizer.param_groups[0]["lr"]}')

            # Save checkpoint
            if checkpoint_dir and (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs - 1) and epoch > 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint(site, model, epoch, checkpoint_dir, args["exp_name"], optimizer)
                save_metrics(checkpoint_dir, epoch, train_metrics, site, split="Train", exp_name=args["exp_name"])

        torch.cuda.empty_cache()

    # -----------------------------
    # Testing mode (model already trained)
    # -----------------------------
    else:
        if not isinstance(model_to_test_path, str):
            raise ValueError("model_to_test_path must be a string.")
        model, loss_fn = load_model_from_path(model_to_test_path, **args)

    # -----------------------------
    # Testing loop
    # -----------------------------
    loader = get_dataloader(site, train=False, test=True, **args)
    if loader.test is not None:
        print("... testing dataloader loaded.")

    y_pred, X, y_true, test_loss, test_metrics = test(model, loss_fn, loader.test, **args)
    print('Test set metrics:', test_metrics)

    results_test = {
        'y_pred': y_pred,
        'y_true': y_true,
        'loss': test_loss,
        'metrics': test_metrics
    }

    # Save test metrics if last of 200 epochs
    if model_to_test_path:
        # regex looks for a number between two underscores (_199_…) right before the file extension .pth
        # example : "checkpoint_199_final.pth" → epoch = 199
        # we hypothesize the model_to_test_path string has as last numerical value between two underscores the 
        # nb of epochs (we saved the models in that format, and 199 corresponds to a model trained on 200 epochs)

        # we only want to save the metrics for the last epoch --> this can be easily changed by changing the if statement bellow
        # need it to save the metrics under a filename containing the number of epochs
        match = re.search(r'_([0-9]+)_\w+\.pth$', model_to_test_path)
        if match:
            epoch = int(match.group(1))
            if epoch == 199:
                save_metrics(checkpoint_dir, epoch, results_test, site, split="Test", exp_name=args["exp_name"])

    return results_test

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--transforms", action="store_true", default=True)
    parser.add_argument("--checkpoint_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_to_test_path", type=str, default=None)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--net", default="densenet", choices=["densenet", "alexnet"])
    parser.add_argument("--nb_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exp_name", type=str, default="densenet121_vbm_bipolar")

    parser.add_argument("--whichsite", default=None,
                        choices=["Baltimore", "Boston", "Dallas", "Detroit", "Hartford",
                                 "mannheim", "creteil", "udine", "galway", "pittsburgh",
                                 "grenoble", "geneve"],
                        help="Run LOSO for a single testing site")
    parser.add_argument("--datasize_idx", type=int, default=8, choices=list(range(9)))

    args = parser.parse_args()

    # Ensure checkpoint directory exists
    create_folder_if_not_exists(args.checkpoint_dir)

    # Build configuration
    config = {
        "checkpoint_dir": None,
        "exp_name": args.exp_name,
        "net": args.net,
        "sampler": "random",
        "nb_epochs": args.nb_epochs,
        "batch_size": args.batch_size,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "transforms": args.transforms,
        "transfer_path": PRETRAINING_MODEL if args.transfer else None,
        "datasize_idx": args.datasize_idx
    }

    # Adjust batch size if transfer mode with specific datasize
    if args.transfer and args.datasize_idx not in [2, 5]:
        config["batch_size"] = 64

    # Logging input params
    print(f"Datasize index: {args.datasize_idx}")
    print(f"Transforms: {config['transforms']}")
    print(f"Epochs: {config['nb_epochs']}")
    print(f"Transfer path: {config['transfer_path']}")
    print(f"Model architecture: {config['net']}")

    # Determine sites to test
    sites = [args.whichsite] if args.whichsite else get_predict_sites_list()
    if args.whichsite:
        print(f"Testing site: {args.whichsite}")

    # Prepare checkpoint directory
    config["checkpoint_dir"] = os.path.join(args.checkpoint_dir, f"n_{args.datasize_idx}/")
    create_folder_if_not_exists(config["checkpoint_dir"])
    print(f"Checkpoint dir: {config['checkpoint_dir']}")

    # Avoid overwriting existing results
    # results_file = os.path.join(config["checkpoint_dir"],
    #                             f"Test_{sites[0]}_{args.exp_name}_epoch{args.nb_epochs-1}.pkl")
    # counter = 1
    # while os.path.exists(results_file):
    #     print(f"{results_file} exists.\nCreating new folder ...")
    #     config["checkpoint_dir"] = f"{config['checkpoint_dir']}_{counter}"
    #     create_folder_if_not_exists(config["checkpoint_dir"])
    #     print(f"New checkpoint dir: {config['checkpoint_dir']}")
    #     results_file = os.path.join(config["checkpoint_dir"],
    #                                 f"Test_{sites[0]}_{args.exp_name}_epoch{args.nb_epochs-1}.pkl")
    #     counter += 1

    # Run experiments
    print(f"Sites to run: {sites}")
    site_results = []

    # if we only wish to test a pretrained model, we must test it on the right LOSO CV test fold (one per model)
    # otherwise, we would have abnormaly high performance metrics due to data leakage (training on a site and then testing on it)
    if args.model_to_test_path: 
        for site in get_predict_sites_list():
            if site in args.model_to_test_path: 
                print("testing only on LOSO site ", site)
                sites = [site]

    for site in sites:
        results_test = run(site=site, model_to_test_path=args.model_to_test_path, **config)
        site_results.append({"site": site, "values": results_test})
        print(f"ROC AUC for site {site}: {results_test['metrics']['roc_auc on test set']}")

    print("All site results:\n", site_results)


if __name__ == '__main__':
    main()
