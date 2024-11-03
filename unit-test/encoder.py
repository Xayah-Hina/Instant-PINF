import torch
import taichi as ti
import numpy as np


class HashEncoderTaichiNerf(torch.nn.Module):
    def __init__(
            self,
            min_res: np.array,
            max_res: np.array,
            num_scales: int,
            max_params=2 ** 19,
            features_per_level: int = 2,
            max_num_queries=10000000,
    ):
        super().__init__()


class HashEncoderHyFluid(torch.nn.Module):
    def __init__(
            self,
            min_res: np.array,
            max_res: np.array,
            num_scales: int,
            max_params=2 ** 19,
            features_per_level: int = 2,
            max_num_queries=10000000,
    ):
        super().__init__()
        b = np.exp((np.log(max_res) - np.log(min_res)) / (num_scales - 1))

        hash_map_shapes = []
        hash_map_sizes = []
        hash_map_indicator = []
        offsets = []
        total_hash_size = 0
        for scale_i in range(num_scales):
            res = np.ceil(min_res * np.power(b, scale_i)).astype(int)
            params_in_level_raw = np.int64(res[0] + 1) * np.int64(res[1] + 1) * np.int64(res[2] + 1) * np.int64(res[3] + 1)
            params_in_level = int(params_in_level_raw) if params_in_level_raw % 8 == 0 else int((params_in_level_raw + 8 - 1) / 8) * 8
            params_in_level = min(max_params, params_in_level)
            hash_map_shapes.append(res)
            hash_map_sizes.append(params_in_level)
            hash_map_indicator.append(1 if params_in_level_raw <= params_in_level else 0)
            offsets.append(total_hash_size)
            total_hash_size += params_in_level * features_per_level

        ####################################################################################################
        self.hash_map_shapes_field = ti.field(dtype=ti.i32, shape=(num_scales, 4))
        self.hash_map_shapes_field.from_numpy(np.array(hash_map_shapes))

        self.hash_map_sizes_field = ti.field(dtype=ti.i32, shape=(num_scales,))
        self.hash_map_sizes_field.from_numpy(np.array(hash_map_sizes))

        self.hash_map_indicator_field = ti.field(dtype=ti.i32, shape=(num_scales,))
        self.hash_map_indicator_field.from_numpy(np.array(hash_map_indicator))

        self.hash_table = torch.nn.Parameter((torch.rand(size=(total_hash_size,), dtype=torch.float32) * 2 - 1) * 0.001, requires_grad=True)

        self.hash_table_field = ti.field(dtype=ti.f32, shape=(total_hash_size,), needs_grad=True)

        self.output_field = ti.field(dtype=ti.f32, shape=(max_num_queries, num_scales * features_per_level), needs_grad=True)
        ####################################################################################################
