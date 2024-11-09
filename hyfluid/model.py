import nerfstudio.models.base_model
import nerfstudio.cameras.cameras
import nerfstudio.cameras.rays
import nerfstudio.data.scene_box
import nerfstudio.field_components.encodings

import torch
import dataclasses
import typing

import encoder


class HyFluidNeRFModelConfig(nerfstudio.models.base_model.ModelConfig):
    _target: typing.Type = dataclasses.field(default_factory=lambda: HyFluidNeRFModel)
    num_coarse_samples: int = 64
    num_importance_samples: int = 128


class HyFluidNeRFModel(nerfstudio.models.base_model.Model):
    config: HyFluidNeRFModelConfig

    def __init__(self, config: HyFluidNeRFModelConfig, scene_box: nerfstudio.data.scene_box.SceneBox, num_train_data: int):
        super().__init__(config=config, scene_box=scene_box, num_train_data=num_train_data)
        self.field_coarse = None
        self.field_fine = None

    def get_param_groups(self) -> typing.Dict[str, typing.List[torch.nn.Parameter]]:
        param_groups = {}
        assert self.field_coarse is not None
        assert self.field_fine is not None
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: typing.Union[nerfstudio.cameras.rays.RayBundle, nerfstudio.cameras.cameras.Cameras]) -> typing.Dict[str, typing.Union[torch.Tensor, typing.List]]:
        pass

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> typing.Dict[str, torch.Tensor]:
        pass

    def get_image_metrics_and_images(self, outputs: typing.Dict[str, torch.Tensor], batch: typing.Dict[str, torch.Tensor]) -> typing.Tuple[typing.Dict[str, float], typing.Dict[str, torch.Tensor]]:
        pass
