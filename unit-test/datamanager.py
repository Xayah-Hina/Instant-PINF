# import nerfstudio.data.datamanagers.base_datamanager
# import nerfstudio.data.dataparsers.base_dataparser
# import nerfstudio.data.datasets.base_dataset
# import nerfstudio.cameras.cameras
# import nerfstudio.cameras.rays
# import nerfstudio.data.scene_box
# import nerfstudio.utils.io
# import nerfstudio.utils.colors
# import nerfstudio.utils.rich_utils
# import nerfstudio.data.utils.dataloaders
# import nerfstudio.data.utils.nerfstudio_collate
# import nerfstudio.data.pixel_samplers
# import nerfstudio.model_components.ray_generators
#
# import imageio.v2 as imageio
# import torch
# import torch.utils.data
# import numpy as np
# import dataclasses
# import pathlib
# import typing
# import os
#
#
# @dataclasses.dataclass
# class HyFluidDataParserConfig(nerfstudio.data.dataparsers.base_dataparser.DataParserConfig):
#     _target: typing.Type = dataclasses.field(default_factory=lambda: HyFluidDataParser)
#     data: pathlib.Path = pathlib.Path("data/ScalarReal")
#     scale_factor: float = 1.0
#     alpha_color: typing.Optional[str] = "white"
#     frame_skip: int = 1
#
#
# @dataclasses.dataclass
# class HyFluidDataParser(nerfstudio.data.dataparsers.base_dataparser.DataParser):
#     config: HyFluidDataParserConfig
#
#     def __init__(self, config: HyFluidDataParserConfig):
#         super().__init__(config=config)
#         if config.alpha_color is not None:
#             self.alpha_color_tensor = nerfstudio.utils.colors.get_color(config.alpha_color)
#         else:
#             self.alpha_color_tensor = None
#
#     def _generate_dataparser_outputs(self, split: str = "train", **kwargs: typing.Optional[typing.Dict]) -> nerfstudio.data.dataparsers.base_dataparser.DataparserOutputs:
#         meta = nerfstudio.utils.io.load_from_json(self.config.data / f"info.json")
#         image_filenames = []
#         all_images = []
#         all_poses = []
#         all_time_steps = []
#         all_focal_lengths = []
#         all_cx = []
#         all_cy = []
#         all_frames = []
#         for video in meta[split + '_videos']:
#             image_array = []
#             pose_array = []
#             time_step_array = []
#             reader = imageio.get_reader(self.config.data / video['file_name'])
#             frame_num = video['frame_num']
#             dt = 1. / frame_num
#             output_folder = self.config.data / pathlib.Path(video['file_name']).stem
#             need_parse_video = True
#             if os.path.exists(output_folder):
#                 need_parse_video = False
#             else:
#                 os.makedirs(output_folder, exist_ok=True)
#             for _idx in range(0, frame_num, self.config.frame_skip):
#                 reader.set_image_index(_idx)
#                 frame = reader.get_next_data()
#                 output_path = output_folder / f'frame_{_idx:04d}.png'
#                 image_filenames.append(output_path)
#                 if need_parse_video:
#                     imageio.imwrite(output_path, frame)
#                 image_array.append(frame)
#                 pose_array.append(video['transform_matrix'])
#                 time_step_array.append(_idx * dt)
#             reader.close()
#             all_images.append((np.array(image_array) / 255.).astype(np.float32))
#             all_poses.append(np.array(pose_array).astype(np.float32))
#             all_time_steps.append(np.array(time_step_array).astype(np.float32))
#             image_height, image_width = all_images[-1].shape[1:3]
#             camera_angle_x = video['camera_angle_x']
#             focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
#             all_focal_lengths.append(float(focal_length))
#             cx = image_width / 2.0
#             cy = image_height / 2.0
#             all_cx.append(float(cx))
#             all_cy.append(float(cy))
#             all_frames.append(frame_num)
#
#         camera_to_world = np.reshape(np.array(all_poses)[:, :, :3, :], (-1, 3, 4))
#         focal_length = np.concatenate([np.repeat(all_focal_lengths[i], all_frames[i]) for i in range(len(all_frames))]).reshape(-1, 1)
#         cx = np.concatenate([np.repeat(all_cx[i], all_frames[i]) for i in range(len(all_frames))]).reshape(-1, 1)
#         cy = np.concatenate([np.repeat(all_cy[i], all_frames[i]) for i in range(len(all_frames))]).reshape(-1, 1)
#         cameras = nerfstudio.cameras.cameras.Cameras(
#             camera_to_worlds=torch.from_numpy(camera_to_world),
#             fx=torch.from_numpy(focal_length),
#             fy=torch.from_numpy(focal_length),
#             cx=torch.from_numpy(cx),
#             cy=torch.from_numpy(cy),
#             camera_type=nerfstudio.cameras.cameras.CameraType.PERSPECTIVE,
#         )
#
#         scene_box = nerfstudio.data.scene_box.SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))
#         metadata = {}
#         dataparser_outputs = nerfstudio.data.dataparsers.base_dataparser.DataparserOutputs(
#             image_filenames=image_filenames,
#             cameras=cameras,
#             alpha_color=self.alpha_color_tensor,
#             scene_box=scene_box,
#             dataparser_scale=self.config.scale_factor,
#             metadata=metadata,
#         )
#
#         return dataparser_outputs
#
#
# @dataclasses.dataclass
# class HyFluidNeRFDataManagerConfig(nerfstudio.data.datamanagers.base_datamanager.DataManagerConfig):
#     _target: typing.Type = dataclasses.field(default_factory=lambda: HyFluidNeRFDataManager)
#     dataparser: HyFluidDataParserConfig = dataclasses.field(default_factory=HyFluidDataParserConfig)
#     train_num_rays_per_batch: int = 1024
#     eval_num_rays_per_batch: int = 1024
#     train_num_images_to_sample_from: int = -1
#     eval_num_images_to_sample_from: int = -1
#     train_num_times_to_repeat_images: int = -1
#     eval_num_times_to_repeat_images: int = -1
#     collate_fn: typing.Callable[[typing.Any], typing.Any] = typing.cast(typing.Any, staticmethod(nerfstudio.data.utils.nerfstudio_collate.nerfstudio_collate))
#     pixel_sampler: nerfstudio.data.pixel_samplers.PixelSamplerConfig = dataclasses.field(default_factory=nerfstudio.data.pixel_samplers.PixelSamplerConfig)
#
#
# class HyFluidNeRFDataManager(nerfstudio.data.datamanagers.base_datamanager.DataManager):
#     config: HyFluidNeRFDataManagerConfig
#     train_dataset: nerfstudio.data.datasets.base_dataset.InputDataset
#     eval_dataset: nerfstudio.data.datasets.base_dataset.InputDataset
#     train_pixel_sampler: nerfstudio.data.pixel_samplers.PixelSampler
#     eval_pixel_sampler: nerfstudio.data.pixel_samplers.PixelSampler
#
#     def __init__(
#             self,
#             config: HyFluidNeRFDataManagerConfig,
#             device: torch.device,
#             test_mode: typing.Literal["test", "val", "inference"] = "val",
#             world_size: int = 1,
#             local_rank: int = 0,
#     ):
#         super().__init__()
#         self.config = config
#         self.device = device
#         self.test_mode = test_mode
#         self.world_size = world_size
#         self.local_rank = local_rank
#
#         self.dataparser: HyFluidDataParser = self.config.dataparser.setup()
#         self.train_dataset = nerfstudio.data.datasets.base_dataset.InputDataset(dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"))
#         self.eval_dataset = nerfstudio.data.datasets.base_dataset.InputDataset(dataparser_outputs=self.dataparser.get_dataparser_outputs(split="test"))
#
#         self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
#         if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
#             self.exclude_batch_keys_from_device.remove("mask")
#         if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
#             self.exclude_batch_keys_from_device.remove("image")
#
#         if self.train_dataset and self.test_mode != "inference":
#             self.train_image_dataloader = nerfstudio.data.utils.dataloaders.CacheDataloader(
#                 self.train_dataset,
#                 num_images_to_sample_from=self.config.train_num_images_to_sample_from,
#                 num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
#                 device=self.device,
#                 num_workers=self.world_size * 4,
#                 pin_memory=True,
#                 collate_fn=self.config.collate_fn,
#                 exclude_batch_keys_from_device=self.train_dataset.exclude_batch_keys_from_device,
#             )
#             self.iter_train_image_dataloader = iter(self.train_image_dataloader)
#             self.train_pixel_sampler = self.config.pixel_sampler.setup(num_rays_per_batch=self.config.train_num_rays_per_batch)
#             self.train_ray_generator = nerfstudio.model_components.ray_generators.RayGenerator(self.train_dataset.cameras.to(self.device))
#         if self.eval_dataset and self.test_mode != "inference":
#             self.eval_image_dataloader = nerfstudio.data.utils.dataloaders.CacheDataloader(
#                 self.eval_dataset,
#                 num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
#                 num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
#                 device=self.device,
#                 num_workers=self.world_size * 4,
#                 pin_memory=True,
#                 collate_fn=self.config.collate_fn,
#                 exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
#             )
#             self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
#             self.eval_pixel_sampler = self.config.pixel_sampler.setup(num_rays_per_batch=self.config.eval_num_rays_per_batch)
#             self.eval_ray_generator = nerfstudio.model_components.ray_generators.RayGenerator(self.eval_dataset.cameras.to(self.device))
#
#     def next_train(self, step: int = 0) -> typing.Tuple[typing.Union[nerfstudio.cameras.rays.RayBundle, nerfstudio.cameras.cameras.Cameras], typing.Dict]:
#         self.train_count += 1
#         image_batch = next(self.iter_train_image_dataloader)
#         batch = self.train_pixel_sampler.sample(image_batch)
#         ray_indices = batch["indices"]
#         ray_bundle = self.train_ray_generator(ray_indices)
#         return ray_bundle, batch
#
#     def next_eval(self, step: int = 0) -> typing.Tuple[typing.Union[nerfstudio.cameras.rays.RayBundle, nerfstudio.cameras.cameras.Cameras], typing.Dict]:
#         self.eval_count += 1
#         image_batch = next(self.iter_eval_image_dataloader)
#         batch = self.eval_pixel_sampler.sample(image_batch)
#         ray_indices = batch["indices"]
#         ray_bundle = self.eval_ray_generator(ray_indices)
#         return ray_bundle, batch
#
#     def next_eval_image(self, step: int) -> typing.Tuple[nerfstudio.cameras.cameras.Cameras, typing.Dict]:
#         pass
#
#     def get_train_rays_per_batch(self) -> int:
#         if self.train_pixel_sampler is not None:
#             return self.train_pixel_sampler.num_rays_per_batch
#         return self.config.train_num_rays_per_batch
#
#     def get_eval_rays_per_batch(self) -> int:
#         if self.eval_pixel_sampler is not None:
#             return self.eval_pixel_sampler.num_rays_per_batch
#         return self.config.eval_num_rays_per_batch
#
#     def get_datapath(self) -> pathlib.Path:
#         return self.config.dataparser.data
#
#     def get_param_groups(self) -> typing.Dict[str, typing.List[torch.nn.Parameter]]:
#         return {}
#
#     def forward(self):
#         pass
#
#     def setup_train(self):
#         pass
#
#     def setup_eval(self):
#         pass
