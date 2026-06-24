"""NuGraph3 event decoder with domain adaptation (MMD)"""
from typing import Any
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sn
from ....util import RecallLoss
from ..types import Data

from ....util.MMDLoss import MMDLoss
from ....util.EmbeddingPlotter import CombinedEmbeddingPlot

class EventDecoderDAmmd(nn.Module):
    """
    NuGraph3 event decoder module for Domain Adaptation on event-level features

    Convolve interaction node embedding down to a set of categorical scores
    for each event class.

    Use MMD loss on interaction node embedding to align them across the source and target dataset and improve classification.

    Args:
        interaction_features: Number of interaction node features
        planes: List of detector planes
        event_classes: List of event classes
    """
    def __init__(self,
                 interaction_features: int,
                 #event_classes: list[str]):  #use this for correct NG3 data where this is listed
                 event_classes: list['cc_nue', 'cc_numu', 'cc_nutau', 'nc'], warmup_epochs: int = 0
                ):
        super().__init__()
        self.warmup_epochs = warmup_epochs  # When to start with domain adaptation.
        self.use_domain_adaptation = False  # Will be updated by main model
        

        # loss functions
        self.loss = RecallLoss()
        self.loss_mmd = MMDLoss()  # Domain Adapttaion - MMD
        mmd = torch.tensor(0.0)

        #OLD: scaling with loss weights directly. Now handeled with temperatures below.
        # self.eta_s = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # self.eta_t = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # self.eta_da = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))        

        # temperature parameters 
        self.tempS = nn.Parameter(torch.tensor(0.))
        self.tempT = nn.Parameter(torch.tensor(0.))
        self.temp_DA = nn.Parameter(torch.tensor(0.))


        # metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": 4 #len(event_classes)  #use this for proper NG3 data  
        }
        self.recall_s = tm.Recall(**metric_args)
        self.precision_s = tm.Precision(**metric_args)
        self.cm_recall_s = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision_s = tm.ConfusionMatrix(normalize="pred", **metric_args)

        self.recall_t = tm.Recall(**metric_args)
        self.precision_t = tm.Precision(**metric_args)
        self.cm_recall_t = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision_t = tm.ConfusionMatrix(normalize="pred", **metric_args)

        self.embeddings = CombinedEmbeddingPlot(method="umap")   # Include latent space plotting 
        
        # network
        self.net = nn.Linear(in_features=interaction_features,
                             out_features=4) #len(event_classes))  #Use 4 for NG3 data
        self.classes = ['cc_nue', 'cc_numu', 'cc_nutau', 'nc']
        #self.classes = event_classes  #use this for proper NG3 data


        

    def forward(self, dataS: Data, dataT: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 event decoder forward pass

        Args:
            dataS: Graph data object (source)
            dataT: Graph data object (target)
            stage: Stage name (train/val/test)
        """

        # run network and calculate loss 
        # data["evt"].x are event features and x are predicted event logits once features are run through "net"; y are true labels
        xS = self.net(dataS["evt"].x)
        yS = dataS["evt"].y
        wS = 2 * (-1 * self.tempS).exp()
        lossS = wS * self.loss(xS, yS) + self.tempS

        xT = self.net(dataT["evt"].x)
        yT = dataT["evt"].y
        wT = 2 * (-1 * self.tempT).exp()
        lossT = wT * self.loss(xT, yT) + self.tempT
        
        if self.use_domain_adaptation:
            """
            Domain Adaptation is implemented via MMD distance (lossDA). 
            Both source and target labels are being used to calculate event losses (lossS and lossT).
            All losses are being scaled via their own temperatures, which are all trainable parameters.
            DA loss is also capped to be at most 1/4 of the lossS (to be on the safe side). 
            """
            #print("Using DA!")

            
            # MMD Alignment
            wDA = 2 * (-1 * self.temp_DA).exp()
            raw_lossDA = wDA * self.loss_mmd(dataS["evt"].x, dataT["evt"].x) + self.temp_DA 

            # Smooth capping of the DA based on the source event loss value (currently to be at most 1/4 of the event loss value)
            sharp = 20.0  # sharpness of transition when source event loss goes from positive to negative
            sig = torch.sigmoid(sharp * lossS)
            max_lossDA = sig * (lossS / 4) + (1 - sig) * (4 * lossS)
            lossDA = torch.min(raw_lossDA, max_lossDA)
            
            # Total loss
            loss = lossS + lossT + lossDA

        else:
            #print("Warmup phase. No event DA!")
            loss = lossS + lossT


        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_event_total/{stage}"] = loss
            metrics[f"recall_event_source/{stage}"] = self.recall_s(xS, yS)
            metrics[f"precision_event_source/{stage}"] = self.precision_s(xS, yS)
            metrics[f"recall_event_target/{stage}"] = self.recall_t(xT, yT)
            metrics[f"precision_event_target/{stage}"] = self.precision_t(xT, yT)
            metrics[f"loss_event_source/{stage}"] = lossS
            metrics[f"loss_event_target/{stage}"] = lossT
            metrics[f"Using_DA_or_not/{stage}"] = int(self.use_domain_adaptation)
            
            if self.use_domain_adaptation:
                metrics[f"DA_loss_capped_event/{stage}"] = lossDA
                metrics[f"DA_loss_uncapped_event/{stage}"] = raw_lossDA 
            
        if stage == "train":
            metrics["temperature/event_source"] = self.tempS
            metrics["temperature/event_target"] = self.tempT
            metrics["temperature/event_DA"] = self.temp_DA
            
        if stage in ["val", "test"]:
            self.cm_recall_s.update(xS, yS)
            self.cm_precision_s.update(xS, yS)
            self.cm_recall_t.update(xT, yT)
            self.cm_precision_t.update(xT, yT)
            self.embeddings.update(dataS["evt"].x, yS, dataT["evt"].x, yT)

        # add inference output to graph object
        dataS["evt"].e = xS.softmax(dim=1)
        if isinstance(dataS, Batch):
            dataS._slice_dict["evt"]["e"] = dataS["evt"].ptr
            incS = torch.zeros(dataS.num_graphs, device=dataS["evt"].x.device)
            dataS._inc_dict["evt"]["e"] = incS

        dataT["evt"].e = xT.softmax(dim=1)
        if isinstance(dataT, Batch):
            dataT._slice_dict["evt"]["e"] = dataT["evt"].ptr
            incT = torch.zeros(dataS.num_graphs, device=dataT["evt"].x.device)
            dataT._inc_dict["evt"]["e"] = incT

        return loss, metrics
        
    
    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw confusion matrix

        Args:
            cm: Confusion matrix object
        """
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel("Assigned label")
        plt.ylabel("True label")
        return fig

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
                     epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
        if not logger:
            return

        logger.experiment.add_figure(f"recall_event_matrix_source/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall_s),
                                     global_step=epoch)
        self.cm_recall_s.reset()

        logger.experiment.add_figure(f"recall_event_matrix_target/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall_t),
                                     global_step=epoch)
        self.cm_recall_t.reset()

        logger.experiment.add_figure(f"precision_event_matrix_source/{stage}",
                                self.draw_confusion_matrix(self.cm_precision_s),
                                global_step=epoch)
        self.cm_precision_s.reset()

        logger.experiment.add_figure(f"precision_event_matrix_target/{stage}",
                                self.draw_confusion_matrix(self.cm_precision_t),
                                global_step=epoch)
        self.cm_precision_t.reset()
        
        # Plot the embedding space 
        dat1, lab1, dat2, lab2 = self.embeddings.compute()
        dat1sub, lab1sub = self.embeddings.subsample(dat1, lab1, max_samples=1000)
        dat2sub, lab2sub = self.embeddings.subsample(dat2, lab2, max_samples=1000)
        
        embeddings_fig = self.embeddings.plot_combined(dat1sub, lab1sub, dat2sub, lab2sub, epoch=epoch, class_names=self.classes)
        logger.experiment.add_figure(f"Embeddings event/{stage}",
                                     embeddings_fig, global_step=epoch)
        self.embeddings.reset()
