from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs

import nerfstudio.utils.io
import imageio.v2 as imageio
import torch
import numpy as np
import dataclasses
import pathlib
import typing


@dataclasses.dataclass
class PINFDataParserConfig(DataParserConfig):
    _target: typing.Type = dataclasses.field(default_factory=lambda: PINFDataParser)
    data: pathlib.Path = pathlib.Path("data/ScalarReal")
    frame_skip: int = 1


@dataclasses.dataclass
class PINFDataParser(DataParser):
    config: PINFDataParserConfig

    def __init__(self, config: PINFDataParserConfig):
        super().__init__(config=config)
        self.data: pathlib.Path = config.data

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: typing.Optional[typing.Dict]):
        meta = nerfstudio.utils.io.load_from_json(self.data / f"info.json")
        all_images = []
        all_poses = []
        all_time_steps = []
        all_focal_lengths = []
        all_cx = []
        all_cy = []
        all_frames = []
        for video in meta[split + '_videos']:
            image_array = []
            pose_array = []
            time_step_array = []
            reader = imageio.get_reader(self.data / video['file_name'])
            frame_num = video['frame_num']
            dt = 1. / frame_num
            for _idx in range(0, frame_num, self.config.frame_skip):
                reader.set_image_index(_idx)
                image_array.append(reader.get_next_data())
                pose_array.append(video['transform_matrix'])
                time_step_array.append(_idx * dt)
            reader.close()
            all_images.append((np.array(image_array) / 255.).astype(np.float32))
            all_poses.append(np.array(pose_array).astype(np.float32))
            all_time_steps.append(np.array(time_step_array).astype(np.float32))
            image_height, image_width = all_images[-1].shape[1:3]
            camera_angle_x = video['camera_angle_x']
            focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
            all_focal_lengths.append(float(focal_length))
            cx = image_width / 2.0
            cy = image_height / 2.0
            all_cx.append(float(cx))
            all_cy.append(float(cy))
            all_frames.append(frame_num)

        camera_to_world = np.reshape(np.array(all_poses)[:, :, :3, :], (-1, 3, 4))
        focal_length = np.concatenate([np.repeat(all_focal_lengths[i], all_frames[i]) for i in range(len(all_frames))]).reshape(-1, 1)
        cx = np.concatenate([np.repeat(all_cx[i], all_frames[i]) for i in range(len(all_frames))]).reshape(-1, 1)
        cy = np.concatenate([np.repeat(all_cy[i], all_frames[i]) for i in range(len(all_frames))]).reshape(-1, 1)
        cameras = Cameras(
            camera_to_worlds=torch.from_numpy(camera_to_world),
            fx=torch.from_numpy(focal_length),
            fy=torch.from_numpy(focal_length),
            cx=torch.from_numpy(cx),
            cy=torch.from_numpy(cy),
            camera_type=CameraType.PERSPECTIVE,
        )


@dataclasses.dataclass
class PINFNeRFDataManagerConfig(DataManagerConfig):
    _target: typing.Type = dataclasses.field(default_factory=lambda: PINFNeRFDataManager)
    dataparser: PINFDataParserConfig = dataclasses.field(default_factory=PINFDataParserConfig)


class PINFNeRFDataManager(DataManager, typing.Generic[TDataset]):
    config: PINFNeRFDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
            self,
            config: PINFNeRFDataManagerConfig,
            device: torch.device,
    ):
        self.config = config
        self.device = device
        super().__init__()

    def forward(self):
        pass

    def setup_train(self):
        pass

    def setup_eval(self):
        pass

    def next_train(self, step: int) -> typing.Tuple[typing.Union[RayBundle, Cameras], typing.Dict]:
        pass

    def next_eval(self, step: int) -> typing.Tuple[typing.Union[RayBundle, Cameras], typing.Dict]:
        pass

    def next_eval_image(self, step: int) -> typing.Tuple[Cameras, typing.Dict]:
        pass

    def get_train_rays_per_batch(self) -> int:
        pass

    def get_eval_rays_per_batch(self) -> int:
        pass

    def get_datapath(self) -> pathlib.Path:
        pass

    def get_param_groups(self) -> typing.Dict[str, typing.List[torch.nn.Parameter]]:
        pass
