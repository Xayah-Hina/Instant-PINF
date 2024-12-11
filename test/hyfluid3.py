import torch
import taichi as ti
import numpy as np
import math

device = torch.device("cuda")

#################################################################################################################################
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


#################################################################################################################################


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


#################################################################################################################################

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


#################################################################################################################################

def get_rays_np_continuous(c2w):
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


def do_resample_rays():
    rays_list = []
    ij = []
    for p in POSES_TRAIN_np[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(p)
        rays_list.append([r_o, r_d])
        ij.append([i_, j_])
    ij = np.stack(ij, 0)
    images_train_sample = sample_bilinear(IMAGE_TRAIN_np, ij)
    ret_IMAGE_TRAIN_gpu = torch.tensor(images_train_sample, device=device, dtype=torch.float32).flatten(start_dim=1,
                                                                                                        end_dim=3)

    rays_np = np.stack(rays_list, 0)
    rays_np = np.transpose(rays_np, [0, 2, 3, 1, 4])
    rays_np = np.reshape(rays_np, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays_np = rays_np.astype(np.float32)
    ret_RAYs_gpu = torch.tensor(rays_np, device=device, dtype=torch.float32)
    ret_RAY_IDX_gpu = torch.randperm(ret_RAYs_gpu.shape[0], device=device, dtype=torch.int32)

    return ret_IMAGE_TRAIN_gpu, ret_RAYs_gpu, ret_RAY_IDX_gpu


#################################################################################################################################

def get_ray_batch(RAYs: torch.Tensor, RAYs_IDX: torch.Tensor, start: int, end: int):
    BATCH_RAYs_IDX = RAYs_IDX[start:end]  # [batch_size]
    BATCH_RAYs_O, BATCH_RAYs_D = torch.transpose(RAYs[BATCH_RAYs_IDX], 0, 1)  # [batch_size, 3]
    return BATCH_RAYs_O, BATCH_RAYs_D, BATCH_RAYs_IDX


def get_frames_at_times(IMAGEs: torch.Tensor, N_frames: int, N_times: int):
    assert N_frames > 1
    TIMEs_IDX = torch.randperm(N_frames, device=IMAGEs.device, dtype=torch.float32)[:N_times] + torch.randn(N_times, device=IMAGEs.device, dtype=torch.float32)  # [N_times]
    TIMEs_IDX_FLOOR = torch.clamp(torch.floor(TIMEs_IDX).long(), 0, N_frames - 1)  # [N_times]
    TIMEs_IDX_CEIL = torch.clamp(torch.ceil(TIMEs_IDX).long(), 0, N_frames - 1)  # [N_times]
    TIMEs_IDX_RESIDUAL = TIMEs_IDX - TIMEs_IDX_FLOOR.float()  # [N_times]
    TIME_STEPs = TIMEs_IDX / (N_frames - 1)  # [N_times]

    FRAMES_INTERPOLATED = IMAGEs[TIMEs_IDX_FLOOR] * (1 - TIMEs_IDX_RESIDUAL).view(-1, 1, 1) + IMAGEs[TIMEs_IDX_CEIL] * TIMEs_IDX_RESIDUAL.view(-1, 1, 1)
    return FRAMES_INTERPOLATED, TIME_STEPs


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


def get_raw(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor):
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

    RAW_FLAT[bbox_mask] = MODEL(ENCODER(POINTS_TIME_FLAT_FINAL))

    # for _ in range(0, POINTS_TIME_FLAT_FINAL.shape[0], batch_size):
    #     RAW_FLAT[bbox_mask][_:_ + batch_size] = MODEL(ENCODER(POINTS_TIME_FLAT_FINAL[_:_ + batch_size]))
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


def get_raw2(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor):
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


if __name__ == '__main__':
    ti.init(arch=ti.cuda, device_memory_GB=36.0)

    pinf_data = np.load("train_dataset.npz")
    IMAGE_TRAIN_np = pinf_data['images_train']
    POSES_TRAIN_np = pinf_data['poses_train']
    HWF_np = pinf_data['hwf']
    RENDER_POSE_np = pinf_data['render_poses']
    RENDER_TIMESTEPs_np = pinf_data['render_timesteps']
    VOXEL_TRAN_np = pinf_data['voxel_tran']
    VOXEL_SCALE_np = pinf_data['voxel_scale']
    NEAR_float = pinf_data['near'].item()
    FAR_float = pinf_data['far'].item()

    from types import SimpleNamespace

    args_npz = np.load("args.npz", allow_pickle=True)
    ARGs = SimpleNamespace(**{
        key: value.item() if isinstance(value, np.ndarray) and value.size == 1 else
        value.tolist() if isinstance(value, np.ndarray) else
        value
        for key, value in args_npz.items()
    })

    from encoder import HashEncoderHyFluid

    ENCODER = HashEncoderHyFluid(
        min_res=np.array([ARGs.base_resolution, ARGs.base_resolution, ARGs.base_resolution, ARGs.base_resolution_t]),
        max_res=np.array(
            [ARGs.finest_resolution, ARGs.finest_resolution, ARGs.finest_resolution, ARGs.finest_resolution_t]),
        num_scales=ARGs.num_levels,
        max_params=2 ** ARGs.log2_hashmap_size).to(device)
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
    ], lr=ARGs.lrate, betas=(0.9, 0.99))
    GRAD_vars += list(ENCODER_params)

    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)

    H = int(HWF_np[0])
    W = int(HWF_np[1])
    FOCAL = float(HWF_np[2])
    K = np.array([[FOCAL, 0, 0.5 * W], [0, FOCAL, 0.5 * H], [0, 0, 1]])

    import tqdm
    batch_size = 256
    time_size = 1
    depth_size = 192

    global_step = 1
    loss_meter = AverageMeter()
    for j in range(1, 2):
        IMAGE_TRAIN_gpu, RAYs_gpu, RAY_IDX_gpu = do_resample_rays()
        for i in tqdm.trange(0, RAY_IDX_gpu.shape[0], batch_size):
            BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, BATCH_RAYs_IDX_gpu = get_ray_batch(RAYs_gpu, RAY_IDX_gpu, i, i + batch_size)  # [batch_size, 3], [batch_size, 3], [batch_size]
            FRAMES_INTERPOLATED_gpu, TIME_STEPs_gpu = get_frames_at_times(IMAGE_TRAIN_gpu, IMAGE_TRAIN_gpu.shape[0], time_size)  # [N_times, N x H x W, 3], [N_times]
            TARGET_S_gpu = FRAMES_INTERPOLATED_gpu[:, BATCH_RAYs_IDX_gpu].flatten(0, 1)  # [batch_size * N_times, 3]
            POINTS_gpu, DISTs_gpu = get_points(BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, NEAR_float, FAR_float, depth_size, randomize=True)  # [batch_size, N_depths, 3]
            for TIME_STEP_gpu in TIME_STEPs_gpu:
                POINTS_TIME_gpu = torch.cat([POINTS_gpu, TIME_STEP_gpu.expand(POINTS_gpu[..., :1].shape)], dim=-1)  # [batch_size, N_depths, 4]
                RGB_MAP = get_raw(POINTS_TIME_gpu, DISTs_gpu, BATCH_RAYs_D_gpu)
                img2mse = lambda x, y: torch.mean((x - y) ** 2)
                img_loss = img2mse(RGB_MAP, TARGET_S_gpu)
                loss = img_loss

                for param in GRAD_vars:  # slightly faster than optimizer.zero_grad()
                    param.grad = None
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item())
                if global_step % 100 == 0:
                    tqdm.tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss_meter.avg:.2g}")
                    loss_meter.reset()

                decay_rate = 0.1
                decay_steps = ARGs.lrate_decay
                new_lrate = ARGs.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                global_step += 1

