import torch
from torch import nn
import torch.nn.functional as F


class simCLR(nn.Module):
    """ Class implementing the simCLR model.
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations
    """
    def __init__(self, args, backbone, return_logits=False):
        super().__init__()
        self.backbone = backbone
        # projector
        sizes = [args.latent_dim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, y1, y2):
        z_i = self.projector(self.backbone(y1))  # shape [N, D]
        z_j = self.projector(self.backbone(y2))  # shape [N, D]
        return z_i, z_j  # letting the loss be computed in yaware_loss.py



class SimCLRFineTunerMinimal_sites(nn.Module):
    def __init__(self, backbone, latent_dim=128, dropout_rate=0.0, nofirst_relu=False, num_sites=12):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.nofirst_relu = nofirst_relu

        # Main task classifier
        self.classifier = nn.Linear(latent_dim, 1)

        # Domain (site) classifier
        self.site_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_sites)   # softmax applied in loss
        )

    def forward(self, x):
        features = self.backbone(x)   # (N, latent_dim)

        if self.nofirst_relu:
            features = F.relu(features, inplace=True)

        features = torch.flatten(features, 1)
        features = self.dropout(features)

        # Task prediction
        task_out = self.classifier(features).squeeze(dim=1)

        # Site prediction
        site_out = self.site_classifier(features)  # logits for sites

        return task_out, site_out


class SimCLRFineTunerMinimal(nn.Module):
    def __init__(self, backbone, latent_dim=128, dropout_rate=0.0, nofirst_relu=True):
        """
            backbone → ReLU → flatten → Dropout → Linear(128→1)

            fine-tuning classifier for SimCLR pretrained encoder.
            Parameters
            ----------
            backbone : nn.Module
                Pretrained encoder (don't freeze - we want to fine-tune it)
            dropout_rate : float
                Dropout probability.
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(latent_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.nofirst_relu = nofirst_relu

    def forward(self, x):
        features = self.backbone(x)  # (N, 128)
        
        if self.nofirst_relu : 
            out = F.relu(features, inplace=True)
            out = torch.flatten(out, 1)
        else : out = torch.flatten(features, 1)
        
        out = self.dropout(out) 
        out = self.classifier(out)  
        
        return out.squeeze(dim=1)

class SimCLRFineTunerPooling(nn.Module):
    def __init__(self, backbone, latent_dim=128, pool_size=8):
        """
        Like in Benoit's SML vs DL
        backbone → ReLU → adaptive avg pool 1D → flatten → Linear(→1)
        
        pool_size: output size after pooling (e.g., 128 → 8 features)
        """
        super().__init__()
        self.backbone = backbone
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_size)
        self.classifier = nn.Linear(pool_size, 1)

    def forward(self, x):
        features = self.backbone(x)  # (N, 128)
        out = F.relu(features, inplace=True)
        
        out = out.unsqueeze(-1)       # (N, 128, 1)
        out = self.adaptive_pool(out)  # (N, pool_size, 1)
        out = torch.flatten(out, 1)   # (N, pool_size)
        
        out = self.classifier(out)    # (N, 1)
        return out.squeeze(dim=1)     # (N,)

class SimCLRFineTunerMinimal_2layersOriginal(nn.Module): # folders named MLP2layers
    def __init__(self, backbone, latent_dim=128, hidden_dim=64, dropout_rate=0.0, nofirst_relu=False):
        """
        backbone → ReLU → flatten → Linear(128→64) → ReLU → (Dropout) → Linear(64→1)

        Fine-tuning classifier for SimCLR pretrained encoder with one hidden layer.
        
        Parameters
        ----------
        backbone : nn.Module
            Pretrained encoder (don't freeze - we want to fine-tune it)
        latent_dim : int
            Output dimension of the backbone encoder
        hidden_dim : int
            Hidden layer dimension
        dropout_rate : float
            Dropout probability.
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_layer = nn.Linear(latent_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.nofirst_relu = nofirst_relu

    def forward(self, x):
        features = self.backbone(x)  # (N, 128)
        if self.nofirst_relu: 
            out = torch.flatten(features, 1)
        else:
            out = F.relu(features, inplace=True)
            out = torch.flatten(out, 1)
        
        # Hidden layer
        out = self.hidden_layer(out)  # (N, hidden_dim)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        # Final classification layer
        out = self.classifier(out)  # (N, 1)
        
        return out.squeeze(dim=1)
    
class SimCLRFineTunerMinimal_2layers(nn.Module): # folders named 2layerMLP
    def __init__(self, backbone, latent_dim=128, hidden_dim=64, dropout_rate=0.0):
        """
        # backbone → ReLU → flatten → Linear(128→64) → ReLU → (Dropout) → Linear(64→1)
        changed to
        backbone  → Linear(128→64) → BatchNorm1d(64) → ReLU → (Dropout) → Linear(64→1)

        Fine-tuning classifier for SimCLR pretrained encoder with one hidden layer.
        
        Parameters
        ----------
        backbone : nn.Module
            Pretrained encoder (don't freeze - we want to fine-tune it)
        latent_dim : int
            Output dimension of the backbone encoder
        hidden_dim : int
            Hidden layer dimension
        dropout_rate : float
            Dropout probability.
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_layer = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) 

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.backbone(x)  # (N, 128)
                
        # Hidden layer
        out = self.hidden_layer(out)  # (N, hidden_dim)
        out = self.batch_norm(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        # Final classification layer
        out = self.classifier(out)  # (N, 1)
        
        return out.squeeze(dim=1)
    

class SimCLRFineTuner(nn.Module):
    def __init__(self, backbone, latent_dim=128, dropout_rate=0.0, last_batch_norm=True):
        """
        backbone → Linear(128→64, no bias) → BatchNorm1d(64) → ReLU → (Dropout) → 
        Linear(64→32, no bias) → BatchNorm1d(32) → ReLU → Linear(32→1)

        :param backbone: pretrained encoder (e.g. SphericalHemiFusionEncoder)
        :param latent_dim: output dimension of the backbone
        :param dropout_rate: dropout probability (if 0.0, no dropout is applied.)
        :param hidden_dim: optional hidden layer dimension. 
        """
        super().__init__()
        self.backbone = backbone
        sizes = [latent_dim] + list(map(int, "64-32".split('-'))) #512-256
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i],sizes[i+1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0.0 and i < len(sizes) - 2:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        if last_batch_norm: layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-1], 1))  
        self.classifier = nn.Sequential(*layers)

            
    def forward(self, x):
        feats = self.backbone(x)          # shape: (N, latent_dim)
        logits = self.classifier(feats)   # shape: (N, 1)
        return logits.squeeze(dim=1)  # shape [N]; for BCEWithLogitsLoss

"""

frozen encoder + SimCLRFineTuner sans dropout : 0.58 pour Baltimore (run 2, 0.5598)
frozen encoder + SimCLRFineTuner avec 0.2 dropout : 0.58 pour Baltimore

frozen encoder + SimCLRFineTuner sans dropout :  0.72 pour Boston
frozen encoder + SimCLRFineTuner avec 0.2 dropout : 0.66 (cata) pour Boston

frozen encoder + SimCLRFineTuner sans dropout :   pour Dallas  0.4710 (run 2, 0.4317)
frozen encoder + SimCLRFineTuner sans dropout :   0.6746 pour Detroit
frozen encoder + SimCLRFineTuner sans dropout :  0.5881 pour Hartford
frozen encoder + SimCLRFineTuner sans dropout :  0.5929 pour creteil (run2, 0.5960)
frozen encoder + SimCLRFineTuner sans dropout :   0.4510 pour galway
frozen encoder + SimCLRFineTuner sans dropout :   pour geneve --> stopped here because clearly it wasn't worth it
frozen encoder + SimCLRFineTuner sans dropout :   pour grenoble
frozen encoder + SimCLRFineTuner sans dropout :   pour mannheim
frozen encoder + SimCLRFineTuner sans dropout :   pour pittsburgh
frozen encoder + SimCLRFineTuner sans dropout :   pour udine

3 min runs for frozen encoder

"""