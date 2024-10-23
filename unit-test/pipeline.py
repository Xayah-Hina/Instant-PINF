from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from nerfstudio.utils import profiler
from torch.nn import Parameter
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dataclasses import field


class PINFNeRFPipelineConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PINFNeRFPipeline)
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


class PINFNeRFPipeline(Pipeline):

    def __init__(self):
        super().__init__()

    def get_train_loss_dict(self, step: int):
        return super().get_train_loss_dict(step)

    def get_eval_loss_dict(self, step: int):
        return super().get_eval_loss_dict(step)

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        super().load_pipeline(loaded_state, step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        pass

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False):
        pass

    @profiler.time_function
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        pass

    @profiler.time_function
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        pass
