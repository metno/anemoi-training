import logging
import math
import os

from anemoi.training.train.forecaster import GraphForecaster
import pytorch_lightning as pl
import torch
from anemoi.models.interface import FuserModelInterface
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)

class NetatmoGraphForecaster(GraphForecaster):

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: tuple,
        metadata: dict,
    ) -> None:
        pl.LigthningModule.__init__()

        graph_data = graph_data.to(self.device)
        
        #TODO use AnemoiModelInterface here if that works
        self.model = FuserModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = tuple(graph_data[mesh].x for mesh in config.graph.) #TODO
        self.node_weights = self.get_node_weights(config, graph_data)

        #TODO
        if config.model.get("output_mask", None) is not None:
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        variable_scaling = self.get_variable_scaling(config, data_indices) #TODO

        _, self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices) #TODO

        loss_kwargs = {"node_weights": self.node_weights}

        scalars = {"variable": (-1, variable_scaling)}

        #TODO
        self.loss = self.get_loss_function(config.training.training_loss, scalars=scalars, **loss_kwargs)

        assert isinstance(self.loss, torch.nn.Module) and not isinstance(
            self.loss,
            torch.nn.ModuleList,
        ), f"Loss function must be a `torch.nn.Module`, not a {type(self.loss).__name__!r}"
        
        #TODO
        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=scalars, **loss_kwargs)
        if not isinstance(self.metrics, torch.nn.ModuleList):
            self.metrics = torch.nn.ModuleList([self.metrics])

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max                

        self.use_zero_optimizer = config.training.zero_optimizer

        self.model_comm_group = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model,
        )





