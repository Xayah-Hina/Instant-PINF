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

from skimage.metrics import structural_similarity


##########################
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


##########################
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


##########################
# Small NeRF for Hash embeddings
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


##########################
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


##########################
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


##########################
def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3, :3]), -1)  # 4.world to 3.target
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    pos_scale = new_pose / (scale_vector)  # 3.target to 2.simulation
    return pos_scale


class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=[0.15, 0.0, 0.15], in_max=[0.85, 1., 0.85]):
        self.s_w2s = torch.tensor(smoke_tran_inv).expand([4, 4]).float()
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = torch.tensor(smoke_scale.copy()).expand([3]).float()
        self.s_min = torch.Tensor(in_min)
        self.s_max = torch.Tensor(in_max)

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


##########################
# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H),
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


##########################
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
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
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
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
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


##########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def render(H, W, K, network_query_fn, model, N_samples, perturb, rays=None, c2w=None,
           near=0., far=1., time_step=None, chunk=1024 * 64):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: float. Focal length of pinhole camera.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    time_step = time_step[:, None, None]  # [N_t, 1, 1]
    N_t = time_step.shape[0]
    N_r = rays.shape[0]
    rays = torch.cat([rays[None].expand(N_t, -1, -1), time_step.expand(-1, N_r, -1)], -1)  # [N_t, n_rays, 7]
    rays = rays.flatten(0, 1)  # [n_time_steps * n_rays, 7]

    # Render and reshape
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(rays[i:i + chunk], network_query_fn=network_query_fn, network_fn=model, N_samples=N_samples, perturb=float(perturb))
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    if N_t == 1:
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = [{k: all_ret[k] for k in all_ret if k not in k_extract}, ]
    return ret_list + ret_dict


def render_path(render_poses, hwf, K, near, far, network_query_fn, model, N_samples, gt_imgs=None, savedir=None, time_steps=None):
    def merge_imgs(save_dir, framerate=30, prefix=''):
        os.system(
            'ffmpeg -hide_banner -loglevel error -y -i {0}/{1}%03d.png -vf palettegen {0}/palette.png'.format(save_dir,
                                                                                                              prefix))
        os.system(
            'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.gif'.format(
                framerate, save_dir, prefix))
        os.system(
            'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.mp4'.format(
                framerate, save_dir, prefix))

    chunk = 512 * 64  # TODO: render_kwargs.update(chunk=512 * 64)
    H, W, focal = hwf
    if time_steps is None:
        time_steps = torch.ones(render_poses.shape[0], dtype=torch.float32)

    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpipss = []

    lpips_net = lpips.LPIPS().cuda()

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, depth, acc, _ = render(H, W, K, network_query_fn = network_query_fn, model = model, N_samples = args.N_samples, perturb = False, c2w=c2w[:3, :4], time_step=time_steps[i][None], chunk=chunk)
        rgbs.append(rgb.cpu().numpy())
        # normalize depth to [0,1]
        depth = (depth - near) / (far - near)
        depths.append(depth.cpu().numpy())

        if gt_imgs is not None:
            gt_img = torch.tensor(gt_imgs[i].squeeze(), dtype=torch.float32)  # [H, W, 3]
            gt_img8 = to8b(gt_img.cpu().numpy())
            gt_img = gt_img[90:960, 45:540]
            rgb = rgb[90:960, 45:540]
            lpips_value = lpips_net(rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1), normalize=True).item()
            p = -10. * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img.cpu().numpy())))
            ssim_value = structural_similarity(gt_img.cpu().numpy(), rgb.cpu().numpy(), data_range=1.0, channel_axis=2)
            lpipss.append(lpips_value)
            psnrs.append(p)
            ssims.append(ssim_value)
            print(f'PSNR: {p:.4g}, SSIM: {ssim_value:.4g}, LPIPS: {lpips_value:.4g}')

        if savedir is not None:
            # save rgb and depth as a figure
            rgb8 = to8b(rgbs[-1])
            imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(i)), rgb8)
            depth = depths[-1]
            colored_depth_map = plt.cm.viridis(depth.squeeze())
            imageio.imwrite(os.path.join(savedir, 'depth_{:03d}.png'.format(i)),
                            (colored_depth_map * 255).astype(np.uint8))

    if savedir is not None:
        merge_imgs(savedir, prefix='rgb_')
        merge_imgs(savedir, prefix='depth_')

    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_lpips = sum(lpipss) / len(lpipss)
        avg_ssim = sum(ssims) / len(ssims)
        print("Avg PSNR over Test set: ", avg_psnr)
        print("Avg LPIPS over Test set: ", avg_lpips)
        print("Avg SSIM over Test set: ", avg_ssim)
        with open(os.path.join(savedir, "test_psnrs_{:0.4f}_lpips_{:0.4f}_ssim_{:0.4f}.json".format(avg_psnr, avg_lpips, avg_ssim)), 'w') as fp:
            json.dump(psnrs, fp)

    return rgbs, depths


def raw2outputs(raw, z_vals, rays_d, learned_rgb=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.ones(3) * (0.6 + torch.tanh(learned_rgb) * 0.4)
    # rgb = 0.6 + torch.tanh(learned_rgb) * 0.4
    noise = 0.

    alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                      :-1]  # [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / (torch.sum(weights, -1) + 1e-10)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)
    depth_map[acc_map < 1e-1] = 0.

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_query_fn,
                network_fn,
                N_samples,
                retraw=False,
                perturb=0.):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    time_step = ray_batch[:, -1]
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    pts_time_step = time_step[..., None, None].expand(-1, pts.shape[1], -1)
    pts = torch.cat([pts, pts_time_step], -1)  # [..., 4]
    pts_flat = torch.reshape(pts, [-1, 4])
    out_dim = 1
    raw_flat = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)

    bbox_mask = bbox_model.insideMask(pts_flat[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True  # in case zero rays are inside the bbox
    pts = pts_flat[bbox_mask]

    raw_flat[bbox_mask] = network_query_fn(pts)
    raw = raw_flat.reshape(N_rays, N_samples, out_dim)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, learned_rgb=network_fn.rgb)

    ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    return ret


if __name__ == '__main__':
    ti.init(arch=ti.cuda, device_memory_GB=36.0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
    hwf = pinf_data_test['hwf']
    render_poses = pinf_data_test['render_poses']
    render_timesteps = pinf_data_test['render_timesteps']
    voxel_tran = pinf_data_test['voxel_tran']
    voxel_scale = pinf_data_test['voxel_scale']
    near = pinf_data_test['near'].item()
    far = pinf_data_test['far'].item()

    render_timesteps = torch.tensor(render_timesteps, device=device, dtype=torch.float32)
    render_poses = torch.tensor(render_poses, device=device, dtype=torch.float32)

    global bbox_model
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # Create nerf model
    ############################
    from encoder import HashEncoderHyFluid

    max_res = np.array([args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
    min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])
    embed_fn = HashEncoderHyFluid(min_res=min_res, max_res=max_res, num_scales=args.num_levels, max_params=2 ** args.log2_hashmap_size)
    input_ch = embed_fn.num_scales * 2  # default 2 params per scale
    embedding_params = list(embed_fn.parameters())

    model = NeRFSmall(num_layers=2,
                      hidden_dim=64,
                      geo_feat_dim=15,
                      num_layers_color=2,
                      hidden_dim_color=16,
                      input_ch=input_ch).to(device)
    print(model)
    print('Total number of trainable parameters in model: {}'.format(
        sum([p.numel() for p in model.parameters() if p.requires_grad])))
    print('Total number of parameters in embedding: {}'.format(
        sum([p.numel() for p in embedding_params if p.requires_grad])))
    grad_vars = list(model.parameters())

    network_query_fn = lambda x: model(embed_fn(x))

    # Create optimizer
    optimizer = RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args.lrate, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    basedir = args.basedir
    expname = args.expname

    ############################

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # For random ray batching
    print('get rays')
    rays = []
    ij = []

    # anti-aliasing
    for p in poses_train[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
        rays.append([r_o, r_d])
        ij.append([i_, j_])
    rays = np.stack(rays, 0)  # [V, ro+rd=2, H, W, 3]
    ij = np.stack(ij, 0)  # [V, 2, H, W]
    images_train = sample_bilinear(images_train_, ij)  # [T, V, H, W, 3]

    rays = np.transpose(rays, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
    rays = np.reshape(rays, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays = rays.astype(np.float32)

    print('done')
    i_batch = 0

    # Move training data to GPU
    images_train = torch.Tensor(images_train).to(device).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
    T, S, _ = images_train.shape
    rays = torch.Tensor(rays).to(device)
    ray_idxs = torch.randperm(rays.shape[0])

    loss_list = []
    psnr_list = []
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    resample_rays = False

    start = 0
    start = start + 1
    global_step = start
    for i in trange(start, args.N_iters + 1):
        # Sample random ray batch
        batch_ray_idx = ray_idxs[i_batch:i_batch + N_rand]
        batch_rays = rays[batch_ray_idx]  # [B, 2, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [2, B, 3]

        i_batch += N_rand
        # temporal bilinear sampling
        time_idx = torch.randperm(T)[:args.N_time].float().to(device)  # [N_t]
        time_idx += torch.randn(args.N_time) - 0.5  # -0.5 ~ 0.5
        time_idx_floor = torch.floor(time_idx).long()
        time_idx_ceil = torch.ceil(time_idx).long()
        time_idx_floor = torch.clamp(time_idx_floor, 0, T - 1)
        time_idx_ceil = torch.clamp(time_idx_ceil, 0, T - 1)
        time_idx_residual = time_idx - time_idx_floor.float()
        frames_floor = images_train[time_idx_floor]  # [N_t, VHW, 3]
        frames_ceil = images_train[time_idx_ceil]  # [N_t, VHW, 3]
        frames_interp = frames_floor * (1 - time_idx_residual).unsqueeze(-1) + frames_ceil * time_idx_residual.unsqueeze(-1)  # [N_t, VHW, 3]
        time_step = time_idx / (T - 1) if T > 1 else torch.zeros_like(time_idx)
        points = frames_interp[:, batch_ray_idx]  # [N_t, B, 3]
        target_s = points.flatten(0, 1)  # [N_t*B, 3]

        if i_batch >= rays.shape[0]:
            print("Shuffle data after an epoch!")
            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = True

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = render(H, W, K, network_query_fn = network_query_fn, model = model, N_samples = args.N_samples, perturb = True, rays=batch_rays, time_step=time_step)

        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        loss_meter.update(loss.item())
        psnr_meter.update(psnr.item())

        for param in grad_vars:  # slightly faster than optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        global_step += 1
        ################################

        # Rest is logging
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            testsavedir = os.path.join(basedir, expname, 'spiral_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(render_poses, hwf, K, near, far, network_query_fn = network_query_fn, model = model, N_samples = args.N_samples, time_steps=render_timesteps, savedir=testsavedir)

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                test_view_pose = torch.tensor(poses_test[0])
                N_timesteps = images_test.shape[0]
                test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
                test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
                render_path(test_view_poses, hwf, K, near, far, network_query_fn = network_query_fn, model = model, N_samples = args.N_samples, time_steps=test_timesteps, gt_imgs=images_test,
                            savedir=testsavedir)

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
            with open(os.path.join(basedir, expname, "loss_vs_time.json"), "w") as fp:
                json.dump(loss_psnr, fp)

        if resample_rays:
            print("Sampling new rays!")
            rays = []
            ij = []
            for p in poses_train[:, :3, :4]:
                r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
                rays.append([r_o, r_d])
                ij.append([i_, j_])
            rays = np.stack(rays, 0)  # [V, ro+rd=2, H, W, 3]
            ij = np.stack(ij, 0)  # [V, 2, H, W]
            images_train = sample_bilinear(images_train_, ij)  # [T, V, H, W, 3]
            rays = np.transpose(rays, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
            rays = np.reshape(rays, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
            rays = rays.astype(np.float32)

            # Move training data to GPU
            images_train = torch.Tensor(images_train).to(device).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
            T, S, _ = images_train.shape
            rays = torch.Tensor(rays).to(device)

            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = False
