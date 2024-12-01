import torch
import taichi as ti
import numpy as np
import imageio.v2 as imageio
import math
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import json
import lpips

device = torch.device("cuda")

if __name__ == '__main__':
    from types import SimpleNamespace

    args_npz = np.load("args.npz", allow_pickle=True)
    args = SimpleNamespace(**{
        key: value.item() if isinstance(value, np.ndarray) and value.size == 1 else
        value.tolist() if isinstance(value, np.ndarray) else
        value
        for key, value in args_npz.items()
    })

    pinf_data = np.load("train_dataset.npz")
    images_train_ = pinf_data['images_train']
    poses_train = pinf_data['poses_train']
    hwf = pinf_data['hwf']
    render_poses = pinf_data['render_poses']
    render_timesteps = pinf_data['render_timesteps']
    voxel_tran = pinf_data['voxel_tran']
    voxel_scale = pinf_data['voxel_scale']
    near = pinf_data['near'].item()
    far = pinf_data['far'].item()

    pinf_data_test = np.load("test_dataset.npz")
    images_test = pinf_data_test['images_test']
    poses_test = pinf_data_test['poses_test']

    ti.init(arch=ti.cuda, device_memory_GB=8.0)
    from encoder import HashEncoderHyFluid

    max_res = np.array([args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
    min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])
    embed_fn = HashEncoderHyFluid(min_res=min_res, max_res=max_res, num_scales=args.num_levels,
                                  max_params=2 ** args.log2_hashmap_size).to(device)
    input_ch = embed_fn.num_scales * 2  # default 2 params per scale
    embedding_params = list(embed_fn.parameters())

    class NeRFSmall(torch.nn.Module):
        def __init__(self,
                     num_layers=3,
                     hidden_dim=64,
                     geo_feat_dim=15,
                     num_layers_color=2,
                     hidden_dim_color=16,
                     input_ch=3,
                     ):
            super(NeRFSmall, self).__init__()

            self.input_ch = input_ch
            self.rgb = torch.nn.Parameter(torch.tensor([0.0], device=device, dtype=torch.float32))

            # sigma network
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.geo_feat_dim = geo_feat_dim

            sigma_net = []
            for l in range(num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = hidden_dim

                if l == num_layers - 1:
                    out_dim = 1  # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim

                sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False, device=device, dtype=torch.float32))

            self.sigma_net = torch.nn.ModuleList(sigma_net)

            self.color_net = []
            for l in range(num_layers_color):
                if l == 0:
                    in_dim = 1
                else:
                    in_dim = hidden_dim_color

                if l == num_layers_color - 1:
                    out_dim = 1
                else:
                    out_dim = hidden_dim_color

                self.color_net.append(torch.nn.Linear(in_dim, out_dim, bias=True, device=device, dtype=torch.float32))

        def forward(self, x):
            h = x
            for l in range(self.num_layers):
                h = self.sigma_net[l](h)
                h = torch.nn.functional.relu(h, inplace=True)

            sigma = h
            return sigma


    model = NeRFSmall(num_layers=2,
                      hidden_dim=64,
                      geo_feat_dim=15,
                      num_layers_color=2,
                      hidden_dim_color=16,
                      input_ch=input_ch).to(device)
    grad_vars = list(model.parameters())

    from torch.optim.optimizer import Optimizer


    class RAdam(Optimizer):

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

            self.degenerated_to_sgd = degenerated_to_sgd
            if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
                for param in params:
                    if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                        param['buffer'] = [[None, None, None] for _ in range(10)]
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                            buffer=[[None, None, None] for _ in range(10)])
            super(RAdam, self).__init__(params, defaults)

        def __setstate__(self, state):
            super(RAdam, self).__setstate__(state)

        def step(self, closure=None):

            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data.float()
                    if grad.is_sparse:
                        raise RuntimeError('RAdam does not support sparse gradients')

                    p_data_fp32 = p.data.float()

                    state = self.state[p]

                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_data_fp32)
                        state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    else:
                        state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    state['step'] += 1
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                        p.data.copy_(p_data_fp32)
                    elif step_size > 0:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                        p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                        p.data.copy_(p_data_fp32)

            return loss


    optimizer = RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args.lrate, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    def pos_world2smoke(Pworld, w2s, scale_vector):
        pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3, :3]), -1)  # 4.world to 3.target
        pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
        new_pose = pos_rot + pos_off
        pos_scale = new_pose / (scale_vector)  # 3.target to 2.simulation
        return pos_scale


    class BBox_Tool(object):
        def __init__(self, smoke_tran_inv, smoke_scale, in_min=[0.15, 0.0, 0.15], in_max=[0.85, 1., 0.85]):
            self.s_w2s = torch.tensor(smoke_tran_inv, device=device, dtype=torch.float32).expand([4, 4])
            self.s2w = torch.inverse(self.s_w2s)
            self.s_scale = torch.tensor(smoke_scale.copy(), device=device, dtype=torch.float32).expand([3])
            self.s_min = torch.tensor(in_min, device=device, dtype=torch.float32)
            self.s_max = torch.tensor(in_max, device=device, dtype=torch.float32)

        def world2sim(self, pts_world):
            pts_world_homo = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
            pts_sim_ = torch.matmul(self.s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
            pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
            return pts_sim

        def world2sim_rot(self, pts_world):
            pts_sim_ = torch.matmul(self.s_w2s[:3, :3], pts_world[..., None]).squeeze(-1)
            pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
            return pts_sim

        def sim2world(self, pts_sim):
            pts_sim_ = pts_sim * self.s_scale
            pts_sim_homo = torch.cat([pts_sim_, torch.ones_like(pts_sim_[..., :1])], dim=-1)
            pts_world = torch.matmul(self.s2w, pts_sim_homo[..., None]).squeeze(-1)[..., :3]
            return pts_world

        def sim2world_rot(self, pts_sim):
            pts_sim_ = pts_sim * self.s_scale
            pts_world = torch.matmul(self.s2w[:3, :3], pts_sim_[..., None]).squeeze(-1)
            return pts_world

        def isInside(self, inputs_pts):
            target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
            above = torch.logical_and(target_pts[..., 0] >= self.s_min[0], target_pts[..., 1] >= self.s_min[1])
            above = torch.logical_and(above, target_pts[..., 2] >= self.s_min[2])
            below = torch.logical_and(target_pts[..., 0] <= self.s_max[0], target_pts[..., 1] <= self.s_max[1])
            below = torch.logical_and(below, target_pts[..., 2] <= self.s_max[2])
            outputs = torch.logical_and(below, above)
            return outputs

        def insideMask(self, inputs_pts, to_float=True):
            return self.isInside(inputs_pts).to(torch.float) if to_float else self.isInside(inputs_pts)


    voxel_tran_inv = np.linalg.inv(voxel_tran)
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale)

    def get_rays_np_continuous(H, W, K, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        random_offset_i = np.random.uniform(0, 1, size=(H, W))
        random_offset_j = np.random.uniform(0, 1, size=(H, W))
        i = i + random_offset_i
        j = j + random_offset_j
        i = np.clip(i, 0, W - 1)
        j = np.clip(j, 0, H - 1)

        dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                        -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d, i, j


    def sample_bilinear(img, xy):
        """
        Sample image with bilinear interpolation
        :param img: (T, V, H, W, 3)
        :param xy: (V, 2, H, W)
        :return: img: (T, V, H, W, 3)
        """
        T, V, H, W, _ = img.shape
        u, v = xy[:, 0], xy[:, 1]

        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        u_floor, v_floor = np.floor(u).astype(int), np.floor(v).astype(int)
        u_ceil, v_ceil = np.ceil(u).astype(int), np.ceil(v).astype(int)

        u_ratio, v_ratio = u - u_floor, v - v_floor
        u_ratio, v_ratio = u_ratio[None, ..., None], v_ratio[None, ..., None]

        bottom_left = img[:, np.arange(V)[:, None, None], v_floor, u_floor]
        bottom_right = img[:, np.arange(V)[:, None, None], v_floor, u_ceil]
        top_left = img[:, np.arange(V)[:, None, None], v_ceil, u_floor]
        top_right = img[:, np.arange(V)[:, None, None], v_ceil, u_ceil]

        bottom = (1 - u_ratio) * bottom_left + u_ratio * bottom_right
        top = (1 - u_ratio) * top_left + u_ratio * top_right

        interpolated = (1 - v_ratio) * bottom + v_ratio * top

        return interpolated


    rays_list = []
    ij = []
    for p in poses_train[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
        rays_list.append([r_o, r_d])
        ij.append([i_, j_])
    rays_np = np.stack(rays_list, 0)  # [V, ro+rd=2, H, W, 3]
    ij = np.stack(ij, 0)  # [V, 2, H, W]
    images_train_sample = sample_bilinear(images_train_, ij)  # [T, V, H, W, 3]

    rays_np = np.transpose(rays_np, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
    rays_np = np.reshape(rays_np, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays_np = rays_np.astype(np.float32)

    images_train_gpu = torch.tensor(images_train_sample, device=device, dtype=torch.float32).flatten(start_dim=1, end_dim=3)
    T, S, _ = images_train_gpu.shape
    rays_gpu = torch.tensor(rays_np, device=device, dtype=torch.float32)
    ray_idxs_gpu = torch.randperm(rays_gpu.shape[0], device=device, dtype=torch.int32)
    print(f'images_train: {images_train_gpu.shape}, rays: {rays_gpu.shape}, T: {T}, S: {S}')

    img2mse = lambda x, y: torch.mean((x - y) ** 2)
    mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=device))
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    class AverageMeter(object):
        """Computes and stores the average and current value"""
        val = 0
        avg = 0
        sum = 0
        count = 0
        tot_count = 0

        def __init__(self):
            self.reset()
            self.tot_count = 0

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = float(val)
            self.sum += float(val) * n
            self.count += n
            self.tot_count += n
            self.avg = self.sum / self.count

    loss_list = []
    psnr_list = []
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    resample_rays = False

    start = 0
    i_batch = 0
    # for i in trange(start + 1, args.N_iters + 1):
    for i in trange(start + 1, start + 2):
        batch_ray_idx = ray_idxs_gpu[i_batch:i_batch + args.N_rand]
        batch_rays = torch.transpose(rays_gpu[batch_ray_idx], 0, 1)

        time_idx = torch.randperm(T, device=device, dtype=torch.float32)[:args.N_time]
        time_idx += torch.randn(args.N_time, device=device, dtype=torch.float32) - 0.5
        time_idx_floor = torch.floor(time_idx).long()
        time_idx_ceil = torch.ceil(time_idx).long()
        time_idx_floor = torch.clamp(time_idx_floor, 0, T - 1)
        time_idx_ceil = torch.clamp(time_idx_ceil, 0, T - 1)
        time_idx_residual = time_idx - time_idx_floor.float()
        frames_floor = images_train_gpu[time_idx_floor]
        frames_ceil = images_train_gpu[time_idx_ceil]
        frames_interp = frames_floor * (1 - time_idx_residual).unsqueeze(-1) + frames_ceil * time_idx_residual.unsqueeze(-1)
        time_step = time_idx / (T - 1) if T > 1 else torch.zeros_like(time_idx)
        points = frames_interp[:, batch_ray_idx]
        target_s = points.flatten(0, 1)

        i_batch += args.N_rand
        if i_batch >= rays_gpu.shape[0]:
            print("Shuffle data after an epoch!")
            ray_idxs_gpu = torch.randperm(rays_gpu.shape[0], device=device)
            i_batch = 0
            resample_rays = True

        #####################################################################################################################################
        _rays_o, _rays_d = batch_rays
        _near_tensor, _far_tensor = near * torch.ones_like(_rays_d[..., :1]), far * torch.ones_like(_rays_d[..., :1])
        _rays = torch.cat([_rays_o, _rays_d, _near_tensor, _far_tensor], -1)

        _t_vals = torch.linspace(0., 1., steps=args.N_samples, device=device, dtype=torch.float32)
        _z_vals = _near_tensor * (1. - _t_vals) + _far_tensor * (_t_vals)

        _mids = .5 * (_z_vals[..., 1:] + _z_vals[..., :-1])
        _upper = torch.cat([_mids, _z_vals[..., -1:]], -1)
        _lower = torch.cat([_z_vals[..., :1], _mids], -1)
        _t_rand = torch.rand(_z_vals.shape, device=device, dtype=torch.float32)
        _z_vals = _lower + (_upper - _lower) * _t_rand

        pts = _rays_o[..., None, :] + _rays_d[..., None, :] * _z_vals[..., :, None]
        time_step_expanded = time_step.expand(pts.shape[0], pts.shape[1], 1)
        pts_with_time = torch.cat([pts, time_step_expanded], dim=-1)
        pts_with_time_flat = torch.reshape(pts_with_time, [-1, pts_with_time.shape[-1]])

        out_dim = 1
        raw_flat = torch.zeros([pts_with_time_flat.shape[0], out_dim], device=device, dtype=torch.float32)

        bbox_mask = bbox_model.insideMask(pts_with_time_flat[..., :3], to_float=False)
        if bbox_mask.sum() == 0:
            bbox_mask[0] = True
        pts_final = pts_with_time_flat[bbox_mask]
        raw_flat[bbox_mask] = model(embed_fn(pts_final))
        raw = raw_flat.reshape(pts_with_time.shape[0], pts_with_time.shape[1], out_dim)

        raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)
        dists = _z_vals[..., 1:] - _z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1)
        dists = dists * torch.norm(_rays_d[..., None, :], dim=-1)
        rgb = torch.ones(3, device=device) * (0.6 + torch.tanh(model.rgb) * 0.4)
        noise = 0.
        alpha = raw2alpha(raw[..., -1] + noise, dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1),
                                        -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * _z_vals, -1) / (torch.sum(weights, -1) + 1e-10)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
        acc_map = torch.sum(weights, -1)
        depth_map[acc_map < 1e-1] = 0.

        ret = {
            'rgb_map': rgb_map,
            'disp_map': disp_map,
            'acc_map': acc_map,
            'weights': weights,
            'depth_map': depth_map,
        }

        #####################################################################################################################################
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        loss_meter.update(loss.item())
        psnr_meter.update(psnr.item())

        for param in grad_vars:  # slightly faster than optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss_meter.avg:.2g}  PSNR: {psnr_meter.avg:.4g}")
            loss_list.append(loss_meter.avg)
            psnr_list.append(psnr_meter.avg)
            loss_psnr = {
                "losses": loss_list,
                "psnr": psnr_list,
            }
            loss_meter.reset()
            psnr_meter.reset()

        if resample_rays:
            print("Sampling new rays!")
            rays_list = []
            ij = []
            for p in poses_train[:, :3, :4]:
                r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
                rays_list.append([r_o, r_d])
                ij.append([i_, j_])
            rays_np = np.stack(rays_list, 0)  # [V, ro+rd=2, H, W, 3]
            ij = np.stack(ij, 0)  # [V, 2, H, W]
            images_train_sample = sample_bilinear(images_train_, ij)  # [T, V, H, W, 3]
            rays_np = np.transpose(rays_np, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
            rays_np = np.reshape(rays_np, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
            rays_np = rays_np.astype(np.float32)

            # Move training data to GPU
            images_train_gpu = torch.tensor(images_train_sample, device=device, dtype=torch.float32).flatten(start_dim=1, end_dim=3)
            T, S, _ = images_train_gpu.shape
            rays_gpu = torch.tensor(rays_np, device=device, dtype=torch.float32)
            ray_idxs_gpu = torch.randperm(rays_gpu.shape[0], device=device, dtype=torch.int32)

            i_batch = 0
            resample_rays = False

