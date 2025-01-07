"""NuGraph3 model architecture"""
import argparse
import warnings
import psutil

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Batch

from pytorch_lightning import LightningModule

from .types import Data
from .encoder import Encoder
from .coreda import NuGraphCoreDA
from .decoders import SemanticDecoder, FilterDecoder, EventDecoder, VertexDecoder, InstanceDecoder

from ...data import H5DataModuleDA

from ...util.MMDLoss import MMDLoss

class NuGraph3DA(LightningModule):
    """
    NuGraph3 model architecture.

    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
        planes: Tuple of planes
        semantic_classes: Tuple of semantic classes
        event_classes: Tuple of event classes
        num_iters: Number of message-passing iterations
        event_head: Whether to enable event decoder
        semantic_head: Whether to enable semantic decoder
        filter_head: Whether to enable filter decoder
        vertex_head: Whether to enable vertex decoder
        use_checkpointing: Whether to use checkpointing
        lr: Learning rate
    """
    def __init__(self,
                 in_features: int = 4,
                 planar_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 instance_features: int = 32,
                 planes: tuple[str] = ('u','v','y'),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 event_classes: tuple[str] = ('numu','nue','nc'),
                 num_iters: int = 5,
                 event_head: bool = True,
                 semantic_head: bool = False,
                 filter_head: bool = False,
                 vertex_head: bool = False,
                 instance_head: bool = False,
                 use_checkpointing: bool = False,
                 lr: float = 0.001):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr

        self.encoder = Encoder(in_features, planar_features,
                               nexus_features, interaction_features,
                               planes)

        self.core_net = NuGraphCoreDA(planar_features,
                                    nexus_features,
                                    interaction_features,
                                    planes,
                                    use_checkpointing)
        self.decoders = []
        ######UPDATED#####
        # self.encoded_batches = []
        # self.encoded_batches_v = []
        #####UPDATED#####

        if event_head:
            self.event_decoder = EventDecoder(
                interaction_features,
                event_classes)
            self.decoders.append(self.event_decoder)
            
        
        if semantic_head:
            self.semantic_decoder = SemanticDecoder(
                planar_features,
                planes,
                semantic_classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(
                planar_features,
                planes,
            )
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(interaction_features)
            self.decoders.append(self.vertex_decoder)

        if instance_head:
            self.instance_decoder = InstanceDecoder(
                planar_features,
                instance_features,
                planes,
            )
            self.decoders.append(self.instance_decoder)

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')

        # metrics
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.

    def forward(self, data: Data,
                stage: str = None):
        """
        NuGraph3 forward function

        This function runs the forward pass of the NuGraph3 architecture,
        and then loops over each decoder to compute the loss and calculate
        and log any performance metrics.

        Args:
            data: Graph data object
            data_t: Graph data object, target domain
            stage: String tag defining the step type
        """
        
        encoding = self.encoder(data)
        for _ in range(self.num_iters):
            encoding = self.core_net(data)

       
        # total_loss = 0.
        # total_metrics = {}
        # for decoder in self.decoders:
        #     loss, metrics = decoder(data, stage)
        #     total_loss += loss
        #     total_metrics.update(metrics)

        # #if hasattr(self, instance_decoder) and self.global_step > 1000:
        # if hasattr(self, "instance_decoder") and self.global_step > 1000:
        #     if isinstance(data, Batch):
        #         data = Batch([self.instance_decoder.materialize(b) for b in data.to_data_list()])
        #     else:
        #         self.instance_decoder.materialize(data)

        # return total_loss, total_metrics
        return encoding

    def on_train_start(self):
        hpmetrics = { 'max_lr': self.hparams.lr }
        self.logger.log_hyperparams(self.hparams, metrics=hpmetrics)
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.

        scalars = {
            'loss': {'loss': [ 'Multiline', [ 'loss/train', 'loss/val' ]]},
            'acc': {}
        }
        for c in self.semantic_classes:
            scalars['acc'][c] = [ 'Multiline', [
                f'semantic_accuracy_class_train/{c}',
                f'semantic_accuracy_class_val/{c}'
            ]]
        self.logger.experiment.add_custom_scalars(scalars)
        
    
    def training_step(self,
                      batch: Data,
                      batch_idx: int, dataloader_idx=0) -> float:

        ###UPDATED#### unpacking batches
        # Unpack the batch (should be a list of tensors)
        if isinstance(batch, list) and len(batch) == 2:
            data1, data2 = batch[0], batch[1] #in case there are no labels just graphs? -seems ok now
        else:
            raise ValueError("Expected batch to be a list containing two DataLoader outputs.")
        
        #####UPDATED######
        loss = 0.
        total_loss = 0.
        decoder_loss = 0.
        total_metrics = {}

        self.encoder(data1)
        for _ in range(self.num_iters):
            encoded1 = self.core_net(data1)

        for decoder in self.decoders:
            loss, metrics = decoder(data1, 'train') 
            loss += loss
            total_metrics.update(metrics)
            
        if hasattr(self, "instance_decoder") and self.global_step > 1000:
            if isinstance(data1, Batch):
                data1 = Batch([self.instance_decoder.materialize(b) for b in data1.to_data_list()])
            else:
                self.instance_decoder.materialize(data1)
                
        self.encoder(data2)
        for _ in range(self.num_iters):
            encoded2 = self.core_net(data2)

        if encoded1 is None:
            raise ValueError("Encoded1 is None. Check the encoder output.")
        if encoded2 is None:
            raise ValueError("Encoded2 is None. Check the encoder output.")
        
        # Calculate MMD Loss
        mmd_loss_instance = MMDLoss()  # Create an instance of MMDLoss
        mmd_loss_value = torch.tensor(0.0)#, device=batch[0].device)  # Initialize MMD loss
        # if len(self.encoded_batches) > 0: # Ensure we have an encoded batch from the first dataloader
        #     last_encoded = self.encoded_batches[-1]
        #     with torch.no_grad(): # Ensure no gradient tracking for the previous batch
        #         mmd_loss_value = mmd_loss_instance(self.encoded_batches[-1], encoded2)
        #    self.log('train_mmd_loss', mmd_loss_value)
        mmd_loss_value = mmd_loss_instance(encoded1, encoded2)
        self.log('train_mmd_loss', mmd_loss_value, batch_size=data1.num_graphs)

        #Update the stored encoded batch
        #self.encoded_batches.append(encoded1)

        # Calculate total loss
        total_loss = loss + 1 * mmd_loss_value  # Scale MMD loss if needed

         # Debug: Print the calculated loss values
        #print(f"Decoder Loss: {loss.item()}, MMD Loss: {mmd_loss_value.item()}, Total Loss: {total_loss.item()}")
        
        self.log('train_total_loss', total_loss)
        #####UPDATED#########
        self.log('loss/train', loss, batch_size=data1.num_graphs, prog_bar=True)
        self.log_dict(metrics, batch_size=data1.num_graphs)
        self.log_memory(data1, 'train')
        return total_loss

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:  
        # Process data from the first validation dataloader
        # Unpack the batch (should be a list of tensors)
        if isinstance(batch, list) and len(batch) == 2:
            data1_v, data2_v = batch[0], batch[1] #in case there are no labels just graphs? -seems ok now
        else:
            raise ValueError("Expected batch to be a list containing two DataLoader outputs.")

        loss_v = 0.
        total_loss_v = 0.
        decoder_loss_v = 0.
        total_metrics_v = {}

        self.encoder(data1_v)
        for _ in range(self.num_iters):
            encoded1_v = self.core_net(data1_v)

        for decoder in self.decoders:
            loss_v, metrics_v = decoder(data1_v, 'val')
            loss_v += loss_v
            total_metrics_v.update(metrics_v)
            
        if hasattr(self, "instance_decoder") and self.global_step > 1000:
            if isinstance(data1_v, Batch):
                data1_v = Batch([self.instance_decoder.materialize(b) for b in data1_v.to_data_list()])
            else:
                self.instance_decoder.materialize(data1_v)

        self.encoder(data2_v)
        for _ in range(self.num_iters):
            encoded2_v = self.core_net(data2_v)
        
        if encoded1_v is None:
            raise ValueError("Encoded1_v is None. Check the encoder output.")
        if encoded2_v is None:
            raise ValueError("Encoded2_v is None. Check the encoder output.")
        
        # Calculate MMD Loss
        mmd_loss_instance_v = MMDLoss()  # Create an instance of MMDLoss
        mmd_loss_value_v = torch.tensor(0.0)#, device=batch[0].device)  # Initialize MMD loss
        # if len(self.encoded_batches_v) > 0: # Ensure we have an encoded batch from the first dataloader
        #     last_encoded = self.encoded_batches_v[-1]
        #     with torch.no_grad(): # Ensure no gradient tracking for the previous batch
        #         mmd_loss_value_v = mmd_loss_instance_v(self.encoded_batches_v[-1], encoded2_v)
        #     self.log('train_mmd_loss_val', mmd_loss_value_v)
        mmd_loss_value_v = mmd_loss_instance_v(encoded1_v, encoded2_v)
        self.log('train_mmd_loss_val', mmd_loss_value_v, batch_size=data1_v.num_graphs)
        
        #Update the stored encoded batch
        #self.encoded_batches_v.append(encoded1_v)

        # Calculate total loss
        total_loss_v = loss_v + 1 * mmd_loss_value_v  # Scale MMD loss if needed

        
        # Debug: Print the calculated loss values
        print(f"Decoder Loss Val: {loss_v.item()}")

        self.log('total_loss_val', loss_v, batch_size=data1_v.num_graphs)
        self.log('loss/val_s', loss_v, batch_size=data1_v.num_graphs)
        self.log_dict(total_metrics_v, batch_size=data1_v.num_graphs)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        
        if isinstance(batch, list) and len(batch) == 2:
            data1_t, data2_t = batch[0], batch[1] #in case there are no labels just graphs? -seems ok now
        else:
            raise ValueError("Expected batch to be a list containing two DataLoader outputs.")
            
        loss_ts, metrics_ts = self(data1_t, 'test', True)
        self.log('loss/test_s', loss_ts, batch_size=data1_t.num_graphs)
        self.log_dict(metrics_t, batch_size=data1_t.num_graphs)
        self.log_memory(batch[0], 'test_source')

        loss_tt, metrics_tt = self(data2_t, 'test', True)
        self.log('loss/test_t', loss_tt, batch_size=data2_t.num_graphs)
        self.log_dict(metrics_tt, batch_size=data2_t.num_graphs)
        self.log_memory(batch[1], 'test_target')

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: Data,
                     batch_idx: int = 0) -> Data:
        
        if isinstance(batch, list) and len(batch) == 2:
            data1_pred, data2_pred = batch[0], batch[1] #in case there are no labels just graphs? -seems ok now
        else:
            raise ValueError("Expected batch to be a list containing two DataLoader outputs.")
            
        pred1 = self(data1_pred)
        pred2 = self(data2_pred)
        return pred1, pred2


    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        onecycle = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    def log_memory(self, batch: Data, stage: str) -> None:
        """
        Log CPU and GPU memory usage

        Args:
            batch: Data object to step over
            stage: String tag defining the step type
        """
        # log CPU memory
        if not hasattr(self, 'max_mem_cpu'):
            self.max_mem_cpu = 0.
        cpu_mem = psutil.Process().memory_info().rss / float(1073741824)
        self.max_mem_cpu = max(self.max_mem_cpu, cpu_mem)
        self.log(f'memory_cpu/{stage}', self.max_mem_cpu,
                 batch_size=batch.num_graphs, reduce_fx=torch.max)

        # log GPU memory
        if not hasattr(self, 'max_mem_gpu'):
            self.max_mem_gpu = 0.
        if self.device != torch.device('cpu'):
            gpu_mem = torch.cuda.memory_reserved(self.device)
            gpu_mem = float(gpu_mem) / float(1073741824)
            self.max_mem_gpu = max(self.max_mem_gpu, gpu_mem)
            self.log(f'memory_gpu/{stage}', self.max_mem_gpu,
                     batch_size=batch.num_graphs, reduce_fx=torch.max)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse argument group for NuGraph3 model

        Args:
            parser: Argument parser to append argument group to
        """
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=4,
                           help='Number of input node features')
        model.add_argument('--planar-feats', type=int, default=128,
                           help='Hidden dimensionality of planar convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--interaction-feats', type=int, default=32,
                           help='Hidden dimensionality of interaction layer')
        model.add_argument('--instance-feats', type=int, default=32,
                           help='Hidden dimensionality of object condensation')
        model.add_argument('--event', action='store_true',
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true',
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true',
                           help='Enable background filter head')
        model.add_argument('--instance', action='store_true',
                           help='Enable instance segmentation head')
        model.add_argument('--vertex', action='store_true',
                           help='Enable vertex regression head')
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModuleDA) -> 'NuGraph3':
        """
        Construct model from arguments

        Args:
            args: Namespace containing parsed arguments
            nudata: Data module
        """
        return cls(
            in_features=args.in_feats,
            planar_features=args.planar_feats,
            nexus_features=args.nexus_feats,
            interaction_features=args.interaction_feats,
            instance_features=args.instance_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            event_classes=nudata.event_classes,
            num_iters=args.num_iters,
            event_head=args.event,
            semantic_head=args.semantic,
            filter_head=args.filter,
            vertex_head=args.vertex,
            instance_head=args.instance,
            use_checkpointing=args.use_checkpointing,
            lr=args.learning_rate)
