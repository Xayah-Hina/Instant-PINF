import torch
import numpy as np
import imageio.v2 as imageio
import nerfstudio.data.scene_box
import nerfstudio.data.dataparsers.base_dataparser
import nerfstudio.data.datasets.base_dataset
import nerfstudio.data.utils.dataloaders
import nerfstudio.data.utils.nerfstudio_collate
import nerfstudio.cameras.cameras
import nerfstudio.utils.io
import nerfstudio.utils.colors

import typing
import pathlib
import os


def load_train_data(dataset_dir: pathlib.Path, split: typing.Literal["train", "val", "test"], device: torch.device, frame_skip: int = 1):
    image_filenames = []
    all_images = []
    all_poses = []
    all_time_steps = []
    all_focal_lengths = []
    all_cx = []
    all_cy = []
    all_frames = []

    meta = nerfstudio.utils.io.load_from_json(dataset_dir / f"info.json")
    for config in meta[split + '_videos']:
        image_array = []
        pose_array = []
        time_step_array = []

        frame_num = config['frame_num']
        dt = 1. / frame_num
        output_folder = dataset_dir / pathlib.Path(config['file_name']).stem
        need_parse_video = True
        if os.path.exists(output_folder):
            need_parse_video = False
        else:
            os.makedirs(output_folder, exist_ok=True)

        reader = imageio.get_reader(dataset_dir / config['file_name'])
        for _idx in range(0, frame_num, frame_skip):
            reader.set_image_index(_idx)
            frame = reader.get_next_data()
            output_image = output_folder / f'frame_{_idx:04d}.png'
            image_filenames.append(output_image)
            if need_parse_video:
                imageio.imwrite(output_image, frame)
            image_array.append(frame)
            pose_array.append(config['transform_matrix'])
            time_step_array.append(_idx * dt)
        reader.close()

        all_images.append((np.array(image_array) / 255.).astype(np.float32))
        all_poses.append(np.array(pose_array).astype(np.float32))
        all_time_steps.append(np.array(time_step_array).astype(np.float32))

        image_height, image_width = all_images[-1].shape[1:3]
        camera_angle_x = config['camera_angle_x']
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
    cameras = nerfstudio.cameras.cameras.Cameras(
        camera_to_worlds=torch.from_numpy(camera_to_world),
        fx=torch.from_numpy(focal_length),
        fy=torch.from_numpy(focal_length),
        cx=torch.from_numpy(cx),
        cy=torch.from_numpy(cy),
        camera_type=nerfstudio.cameras.cameras.CameraType.PERSPECTIVE,
    ).to(device)

    scene_box = nerfstudio.data.scene_box.SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))
    metadata = {'all_frames': all_frames}
    alpha_color_tensor = nerfstudio.utils.colors.get_color("white")
    dataparser_outputs = nerfstudio.data.dataparsers.base_dataparser.DataparserOutputs(
        image_filenames=image_filenames,
        cameras=cameras,
        alpha_color=alpha_color_tensor,
        scene_box=scene_box,
        dataparser_scale=1.0,
        metadata=metadata,
    )

    dataset = nerfstudio.data.datasets.base_dataset.InputDataset(dataparser_outputs=dataparser_outputs)
    dataloader = nerfstudio.data.utils.dataloaders.CacheDataloader(
        dataset=dataset,
        device=device,
        num_workers=16,
        collate_fn=nerfstudio.data.utils.nerfstudio_collate.nerfstudio_collate,
        exclude_batch_keys_from_device=[],
    )

    return dataloader


def create_test(dataset: nerfstudio.data.datasets.base_dataset.InputDataset, device: torch.device):
    dataloader = nerfstudio.data.utils.dataloaders.FixedIndicesEvalDataloader(input_dataset=dataset, device=device)
    camera, batch = next(dataloader)
    camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    num_rays = len(camera_ray_bundle)

    print(f'Number of rays: {num_rays}')


if __name__ == '__main__':
    import time

    start_time = time.time()
    dataloader = load_train_data(pathlib.Path("C:/Users/imeho/Documents/DataSets/InstantPINF/ScalarReal"), "train", device=torch.device("cuda"))
    end_time = time.time()
    print(f'Time to load data: {end_time - start_time:.2f} seconds')

    iter = iter(dataloader)
    image_batch = next(iter)

    print(f'image device: {image_batch["image"].device}')
    memory_image_cuda = image_batch['image'].element_size() * image_batch['image'].numel()
    print(f'Memory of image: {memory_image_cuda / 1024 / 1024:.2f} MB')
