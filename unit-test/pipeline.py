import nerfstudio.pipelines.base_pipeline
import nerfstudio.configs.base_config
import nerfstudio.data.datamanagers.base_datamanager
import nerfstudio.models.base_model
import nerfstudio.engine.callbacks
from nerfstudio.utils import profiler

import torch
import torch.nn
import dataclasses
import typing
import pathlib


class PINFNeRFPipelineConfig(nerfstudio.configs.base_config.InstantiateConfig):
    _target: typing.Type = dataclasses.field(default_factory=lambda: PINFNeRFPipeline)
    datamanager: nerfstudio.data.datamanagers.base_datamanager.DataManagerConfig = dataclasses.field(default_factory=nerfstudio.data.datamanagers.base_datamanager.DataManagerConfig)
    model: nerfstudio.models.base_model.ModelConfig = dataclasses.field(default_factory=nerfstudio.models.base_model.ModelConfig)


class PINFNeRFPipeline(nerfstudio.pipelines.base_pipeline.Pipeline):
    def __init__(
            self,
            config: PINFNeRFPipelineConfig,
            device: str,
            test_mode: typing.Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.world_size = world_size
        self.local_rank = local_rank

        self.datamanager = self.config.datamanager.setup(device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank)

    def get_train_loss_dict(self, step: int):
        return super().get_train_loss_dict(step)

    def get_eval_loss_dict(self, step: int):
        return super().get_eval_loss_dict(step)

    def load_pipeline(self, loaded_state: typing.Dict[str, typing.Any], step: int) -> None:
        super().load_pipeline(loaded_state, step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        pass

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: typing.Optional[int] = None, output_path: typing.Optional[pathlib.Path] = None, get_std: bool = False):
        pass

    @profiler.time_function
    def get_training_callbacks(self, training_callback_attributes: nerfstudio.engine.callbacks.TrainingCallbackAttributes) -> typing.List[nerfstudio.engine.callbacks.TrainingCallback]:
        pass

    @profiler.time_function
    def get_param_groups(self) -> typing.Dict[str, typing.List[torch.nn.Parameter]]:
        pass
