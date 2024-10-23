from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle

import torch
import dataclasses
import typing


class PINFNeRFModelConfig(ModelConfig):
    _target: typing.Type = dataclasses.field(default_factory=lambda: PINFNeRFModel)


class PINFNeRFModel(Model):
    def __init__(self):
        super().__init__()

    def get_param_groups(self) -> typing.Dict[str, typing.List[torch.nn.Parameter]]:
        pass

    def get_outputs(self, ray_bundle: typing.Union[RayBundle, Cameras]) -> typing.Dict[str, typing.Union[torch.Tensor, typing.List]]:
        pass

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> typing.Dict[str, torch.Tensor]:
        pass

    def get_image_metrics_and_images(self, outputs: typing.Dict[str, torch.Tensor], batch: typing.Dict[str, torch.Tensor]) -> typing.Tuple[typing.Dict[str, float], typing.Dict[str, torch.Tensor]]:
        pass
