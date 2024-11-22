import logging
import math
import os

from anemoi.training.train.forecaster import GraphForecaster
import pytorch_lightning as pl
import torch
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData
from torch.distributed.distributed_c10d import ProcessGroup


from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)

class NetatmoGraphForecaster(pl.LightningModule):

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: tuple,
        metadata: dict,
    ) -> None:
        super().__init__()

        graph_data = graph_data.to(self.device)
        '''
        #TODO use AnemoiModelInterface here if that works
        self.model = AnemoiModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )
        '''
        self.data_indices = data_indices

        self.save_hyperparameters()

#        self.latlons_data = tuple(graph_data[mesh].x for mesh in config.graph.) #TODO
        self.node_weights = self.get_node_weights(config, graph_data) #TODO

        #TODO
        if config.model.get("output_mask", None) is not None:
            raise NotImplementedError("output mask not supported in NetatmoGraphForecaster")
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        variable_scaling = self.get_variable_scaling(config, data_indices) #TODO

        _, self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices) #TODO

        loss_kwargs = tuple({"node_weights": node_weights} for node_weights in self.node_weights)

        scalars = tuple({"variable": (-1, scaling)} for scaling in variable_scaling)

        self.loss = torch.nn.ModuleList(
            [GraphForecaster.get_loss_function(
                loss_config, 
                scalars=scalars[i],
                **loss_kwargs[i], 
            )
            for i, loss_config in enumerate(config.training.training_loss)
            ],
        )

        #TODO
#        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=scalars, **loss_kwargs)

        if config.training.loss_gradient_scaling:
            raise NotImplementedError("Loss gradient scaling not available for NetatmoGraphForecaster")
#            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

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

    def forward(self, x: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        return self.model(x, self.model_comm_group)

    @staticmethod
    def get_val_metric_ranges(
        config: DictConfig, 
        data_indices: tuple
    ) -> tuple[dict, dict]:
        return tuple(GraphForecaster.get_val_metric_ranges(config, data_index) for data_index in data_indices)

    @staticmethod
    def get_variable_scaling(
        config: DictConfig,
        data_indices: tuple
    ) -> tuple:
        return tuple(GraphForecaster.get_variable_scaling(config, data_index) for data_index in data_indices)

    @staticmethod
    def get_node_weights(
        config: DictConfig,
        graph_data: HeteroData
    ) -> tuple:
        print("get_node_weights not implemented")
        return (torch.tensor([0., 0.]), torch.tensor([0., 0.]))

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    #TODO: Figure out what to do with self.output_mask
    def advance_input(
        self,
        x: tuple[torch.Tensor],
        y_pred: tuple[torch.Tensor],
        batch: tuple[torch.Tensor],
        rollout_step: int,
    ) -> tuple[torch.Tensor]:
        
        for x_elem, y_elem, batch_elem, data_indices in zip(x, y_pred, batch, self.data_indices):
            x_elem = x_elem.roll(-1, dims=1)
            x_elem[:, -1, :, :, data_indices.internal_model.input.prognostic] = y_elem[
            ...,
            data_indices.internal_model.output.prognostic,
            ]

            x_elem[:, -1] = self.output_mask.rollout_boundary(x_elem[:, -1], batch_elem[:, -1], data_indices)

            # get new "constants" needed for time-varying fields
            x_elem[:, -1, :, :, data_indices.internal_model.input.forcing] = batch_elem[
                :,
                self.multi_step + rollout_step,
                :,
                :,
                data_indices.internal_data.input.forcing,
            ]
        return x
    