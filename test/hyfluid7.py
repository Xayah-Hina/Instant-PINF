import torch
import numpy as np
import imageio.v2 as imageio
import taichi as ti
import tqdm
import os

device = torch.device("cuda")


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device),
                          indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_points(RAYs_O: torch.Tensor, RAYs_D: torch.Tensor, near: float, far: float, N_depths: int, randomize: bool):
    T_VALs = torch.linspace(0., 1., steps=N_depths, device=RAYs_D.device, dtype=torch.float32)  # [N_depths]
    Z_VALs = near * torch.ones_like(RAYs_D[..., :1]) * (1. - T_VALs) + far * torch.ones_like(RAYs_D[..., :1]) * T_VALs  # [batch_size, N_depths]

    if randomize:
        MID_VALs = .5 * (Z_VALs[..., 1:] + Z_VALs[..., :-1])  # [batch_size, N_depths-1]
        UPPER_VALs = torch.cat([MID_VALs, Z_VALs[..., -1:]], -1)  # [batch_size, N_depths]
        LOWER_VALs = torch.cat([Z_VALs[..., :1], MID_VALs], -1)  # [batch_size, N_depths]
        T_RAND = torch.rand(Z_VALs.shape, device=RAYs_D.device, dtype=torch.float32)  # [batch_size, N_depths]
        Z_VALs = LOWER_VALs + (UPPER_VALs - LOWER_VALs) * T_RAND  # [batch_size, N_depths]

    DIST_VALs = Z_VALs[..., 1:] - Z_VALs[..., :-1]  # [batch_size, N_depths-1]
    POINTS = RAYs_O[..., None, :] + RAYs_D[..., None, :] * Z_VALs[..., :, None]  # [batch_size, N_depths, 3]
    return POINTS, DIST_VALs


def get_raw2(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor, BBOX_MODEL_gpu, MODEL, ENCODER):
    assert POINTS_TIME.dim() == 3 and POINTS_TIME.shape[-1] == 4
    assert POINTS_TIME.shape[0] == DISTs.shape[0] == RAYs_D_FLAT.shape[0]
    POINTS_TIME_FLAT = POINTS_TIME.view(-1, POINTS_TIME.shape[-1])  # [batch_size * N_depths, 4]
    out_dim = 1
    RAW_FLAT = torch.zeros([POINTS_TIME_FLAT.shape[0], out_dim], device=POINTS_TIME_FLAT.device, dtype=torch.float32)
    bbox_mask = BBOX_MODEL_gpu.insideMask(POINTS_TIME_FLAT[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True
        assert False
    POINTS_TIME_FLAT_FINAL = POINTS_TIME_FLAT[bbox_mask]

    chunk = 512 * 64
    ret_list = []
    for _ in range(0, POINTS_TIME_FLAT_FINAL.shape[0], chunk):
        ret = MODEL(ENCODER(POINTS_TIME_FLAT_FINAL[_:min(_ + chunk, POINTS_TIME_FLAT_FINAL.shape[0])]))
        ret_list.append(ret)
    all_ret = torch.cat(ret_list, 0)
    RAW_FLAT[bbox_mask] = all_ret
    RAW = RAW_FLAT.reshape(*POINTS_TIME.shape[:-1], out_dim)
    assert RAW.dim() == 3 and RAW.shape[-1] == 1

    DISTs_cat = torch.cat([DISTs, torch.tensor([1e10], device=DISTs.device).expand(DISTs[..., :1].shape)], -1)  # [batch_size, N_depths]
    DISTS_final = DISTs_cat * torch.norm(RAYs_D_FLAT[..., None, :], dim=-1)  # [batch_size, N_depths]

    RGB_TRAINED = torch.ones(3, device=POINTS_TIME.device) * (0.6 + torch.tanh(MODEL.rgb) * 0.4)
    raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)
    noise = 0.
    alpha = raw2alpha(RAW[..., -1] + noise, DISTS_final)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * RGB_TRAINED, -2)
    return rgb_map


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
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

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

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))

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

            self.color_net.append(torch.nn.Linear(in_dim, out_dim, bias=True))

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, inplace=True)

        sigma = h
        return sigma


from torch.optim.optimizer import Optimizer
import math


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


if __name__ == '__main__':
    ti.init(arch=ti.cuda, device_memory_GB=36.0)
    ###################################################################################
    pinf_data_test = np.load("test_dataset.npz")
    IMAGE_TEST_np = pinf_data_test['images_test']
    POSES_TEST_np = pinf_data_test['poses_test']
    HWF_np = pinf_data_test['hwf']
    VOXEL_TRAN_np = pinf_data_test['voxel_tran']
    VOXEL_SCALE_np = pinf_data_test['voxel_scale']
    NEAR_float = pinf_data_test['near'].item()
    FAR_float = pinf_data_test['far'].item()
    H = int(HWF_np[0])
    W = int(HWF_np[1])
    FOCAL = float(HWF_np[2])
    K = np.array([[FOCAL, 0, 0.5 * W], [0, FOCAL, 0.5 * H], [0, 0, 1]])

    test_view_pose = torch.tensor(POSES_TEST_np[0], device=device, dtype=torch.float32)
    N_timesteps = IMAGE_TEST_np.shape[0]
    test_timesteps = torch.arange(N_timesteps, device=device) / (N_timesteps - 1)
    rays_o, rays_d = get_rays(H, W, K, test_view_pose)
    points, dists = get_points(rays_o, rays_d, NEAR_float, FAR_float, 192, randomize=False)

    points_flat = points.flatten(0, 1)
    dists_flat = dists.flatten(0, 1)
    rays_d_flat = rays_d.flatten(0, 1)
    ###################################################################################
    base_resolution = 16
    base_resolution_t = 16
    finest_resolution = 256
    finest_resolution_t = 128
    num_levels = 16
    log2_hashmap_size = 19
    lrate = 0.01

    from encoder import HashEncoderHyFluid

    ENCODER = HashEncoderHyFluid(
        min_res=np.array([base_resolution, base_resolution, base_resolution, base_resolution_t]),
        max_res=np.array(
            [finest_resolution, finest_resolution, finest_resolution, finest_resolution_t]),
        num_scales=num_levels,
        max_params=2 ** log2_hashmap_size).to(device)
    ENCODER_params = list(ENCODER.parameters())

    MODEL = NeRFSmall(num_layers=2,
                      hidden_dim=64,
                      geo_feat_dim=15,
                      num_layers_color=2,
                      hidden_dim_color=16,
                      input_ch=ENCODER.num_scales * 2).to(device)
    GRAD_vars = list(MODEL.parameters())

    optimizer = RAdam([
        {'params': GRAD_vars, 'weight_decay': 1e-6},
        {'params': ENCODER_params, 'eps': 1e-15}
    ], lr=lrate, betas=(0.9, 0.99))
    GRAD_vars += list(ENCODER_params)

    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)
    ###################################################################################
    ckpts = [os.path.join("checkpoint", f) for f in sorted(os.listdir("checkpoint")) if 'tar' in f]
    print(ckpts)
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path)
    MODEL.load_state_dict(ckpt['network_fn_state_dict'])
    ENCODER.load_state_dict(ckpt['embed_fn_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    ###################################################################################
    with torch.no_grad():
        for i in tqdm.trange(0, test_timesteps.shape[0]):
            test_timesteps_expended = test_timesteps[i].expand(points_flat[..., :1].shape)
            points_time_flat_gpu = torch.cat([points_flat, test_timesteps_expended], dim=-1)
            rgb_map_flat = get_raw2(points_time_flat_gpu, dists_flat, rays_d_flat, BBOX_MODEL_gpu, MODEL, ENCODER)
            rgb_map = rgb_map_flat.view(points.shape[0], points.shape[1], rgb_map_flat.shape[-1])
            to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
            rgb8 = to8b(rgb_map.cpu().numpy())
            imageio.imsave(os.path.join("output", 'rgb_{:03d}.png'.format(i)), rgb8)
