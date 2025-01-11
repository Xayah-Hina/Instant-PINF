import src.radam as radam
import src.model as mmodel

import numpy as np
import torch
from tqdm import tqdm, trange
import os
import json
############################################################################################################
# from taichi_encoders.mgpcg import MGPCG_3

import taichi as ti
import math
import time


@ti.func
def sample(qf: ti.template(), u: float, v: float, w: float):
    u_dim, v_dim, w_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim - 1))
    j = ti.max(0, ti.min(int(v), v_dim - 1))
    k = ti.max(0, ti.min(int(w), w_dim - 1))
    return qf[i, j, k]


@ti.kernel
def split_central_vector(vc: ti.template(), vx: ti.template(), vy: ti.template(), vz: ti.template()):
    for i, j, k in vx:
        r = sample(vc, i, j, k)
        l = sample(vc, i - 1, j, k)
        vx[i, j, k] = 0.5 * (r.x + l.x)
    for i, j, k in vy:
        t = sample(vc, i, j, k)
        b = sample(vc, i, j - 1, k)
        vy[i, j, k] = 0.5 * (t.y + b.y)
    for i, j, k in vz:
        c = sample(vc, i, j, k)
        a = sample(vc, i, j, k - 1)
        vz[i, j, k] = 0.5 * (c.z + a.z)


@ti.kernel
def get_central_vector(vx: ti.template(), vy: ti.template(), vz: ti.template(), vc: ti.template()):
    for i, j, k in vc:
        vc[i, j, k].x = 0.5 * (vx[i + 1, j, k] + vx[i, j, k])
        vc[i, j, k].y = 0.5 * (vy[i, j + 1, k] + vy[i, j, k])
        vc[i, j, k].z = 0.5 * (vz[i, j, k + 1] + vz[i, j, k])


@ti.data_oriented
class MGPCG:
    '''
Grid-based MGPCG solver for the possion equation.

.. note::

    This solver only runs on CPU and CUDA backends since it requires the
    ``pointer`` SNode.
    '''

    def __init__(self, boundary_types, N, dim=2, base_level=3, real=float):
        '''
        :parameter dim: Dimensionality of the fields.
        :parameter N: Grid resolutions.
        :parameter n_mg_levels: Number of multigrid levels.
        '''

        # grid parameters
        self.use_multigrid = True

        self.N = N
        self.n_mg_levels = int(math.log2(min(N))) - base_level + 1
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.dim = dim
        self.real = real

        # setup sparse simulation data arrays
        self.r = [ti.field(dtype=self.real)
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(dtype=self.real)
                  for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real)  # step size
        self.beta = ti.field(dtype=self.real)  # step size
        self.sum = ti.field(dtype=self.real)  # storage for reductions
        self.r_mean = ti.field(dtype=self.real)  # storage for avg of r
        self.num_entries = math.prod(self.N)

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [n // 4 for n in self.N]).dense(
            indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [n // (4 * 2 ** l) for n in self.N]).dense(
                indices,
                4).place(self.r[l], self.z[l])

        ti.root.place(self.alpha, self.beta, self.sum, self.r_mean)

        self.boundary_types = boundary_types

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        '''
        Set up the solver for $\nabla^2 x = k r$, a scaled Poisson problem.
        :parameter k: (scalar) A scaling factor of the right-hand side.
        :parameter r: (ti.field) Unscaled right-hand side.
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            self.init_r(I, r[I] * k)

    @ti.kernel
    def get_result(self, x: ti.template()):
        '''
        Get the solution field.

        :parameter x: (ti.field) The field to store the solution
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            x[I] = self.x[I]

    @ti.func
    def neighbor_sum(self, x, I):
        dims = x.shape
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # add right if has right
            if I[i] < dims[i] - 1:
                ret += x[I + offset]
            # add left if has left
            if I[i] > 0:
                ret += x[I - offset]
        return ret

    @ti.func
    def num_fluid_neighbors(self, x, I):
        dims = x.shape
        num = 2.0 * self.dim
        for i in ti.static(range(self.dim)):
            if I[i] <= 0 and self.boundary_types[i, 0] == 2:
                num -= 1.0
            if I[i] >= dims[i] - 1 and self.boundary_types[i, 1] == 2:
                num -= 1.0
        return num

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            multiplier = self.num_fluid_neighbors(self.p, I)
            self.Ap[I] = multiplier * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def get_Ap(self, p: ti.template(), Ap: ti.template()):
        for I in ti.grouped(Ap):
            multiplier = self.num_fluid_neighbors(p, I)
            Ap[I] = multiplier * p[I] - self.neighbor_sum(
                p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            multiplier = self.num_fluid_neighbors(self.z[l], I)
            res = self.r[l][I] - (multiplier * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 1.0 / (self.dim - 1.0)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                multiplier = self.num_fluid_neighbors(self.z[l], I)
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / multiplier

    @ti.kernel
    def recenter(self, r: ti.template()):  # so that the mean value of r is 0
        self.r_mean[None] = 0.0
        for I in ti.grouped(r):
            self.r_mean[None] += r[I] / self.num_entries
        for I in ti.grouped(r):
            r[I] -= self.r_mean[None]

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,
              max_iters=-1,
              eps=1e-12,
              tol=1e-12,
              verbose=False):
        '''
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        '''
        all_neumann = (self.boundary_types.sum() == 2 * 2 * self.dim)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p

        if all_neumann:
            self.recenter(self.r[0])
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]
        # print("[MGPCG] Starting error: ", math.sqrt(old_zTr))

        # Conjugate gradients
        it = 0
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f'iter {it}, |residual|_2={math.sqrt(rTr)}')

            if rTr < tol:
                end_t = time.time()
                # print("[MGPCG] final error: ", math.sqrt(rTr), " using time: ", end_t - start_t)
                return

            if all_neumann:
                self.recenter(self.r[0])
            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            it += 1

        end_t = time.time()
        # print("[MGPCG] Return without converging at iter: ", it, " with final error: ", math.sqrt(rTr), " using time: ",
        #       end_t - start_t)


class MGPCG_3(MGPCG):

    def __init__(self, boundary_types, N, base_level=3, real=float):
        super().__init__(boundary_types, N, dim=3, base_level=base_level, real=real)

        rx, ry, rz = N
        self.u_div = ti.field(float, shape=N)
        self.p = ti.field(float, shape=N)
        self.boundary_types = boundary_types
        self.u_x = ti.field(float, shape=(rx + 1, ry, rz))
        self.u_y = ti.field(float, shape=(rx, ry + 1, rz))
        self.u_z = ti.field(float, shape=(rx, ry, rz + 1))
        self.u = ti.Vector.field(3, float, shape=(rx, ry, rz))
        self.u_y_bottom = ti.field(float, shape=(rx, 1, rz))

    @ti.kernel
    def apply_bc(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in u_x:
            if i == 0 and self.boundary_types[0, 0] == 2:
                u_x[i, j, k] = 0
            if i == u_dim - 1 and self.boundary_types[0, 1] == 2:
                u_x[i, j, k] = 0
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in u_y:
            if j == 0 and self.boundary_types[1, 0] == 2:
                u_y[i, j, k] = self.u_y_bottom[i, j, k]
                # u_y[i, j, k] = 0.5
            if j == v_dim - 1 and self.boundary_types[1, 1] == 2:
                u_y[i, j, k] = 0
        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in u_z:
            if k == 0 and self.boundary_types[2, 0] == 2:
                u_z[i, j, k] = 0
            if k == w_dim - 1 and self.boundary_types[2, 1] == 2:
                u_z[i, j, k] = 0

    @ti.kernel
    def divergence(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.u_div.shape
        for i, j, k in self.u_div:
            vl = sample(u_x, i, j, k)
            vr = sample(u_x, i + 1, j, k)
            vb = sample(u_y, i, j, k)
            vt = sample(u_y, i, j + 1, k)
            va = sample(u_z, i, j, k)
            vc = sample(u_z, i, j, k + 1)
            self.u_div[i, j, k] = vr - vl + vt - vb + vc - va

    @ti.kernel
    def subtract_grad_p(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.p.shape
        for i, j, k in u_x:
            pr = sample(self.p, i, j, k)
            pl = sample(self.p, i - 1, j, k)
            if i - 1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_x[i, j, k] -= (pr - pl)
        for i, j, k in u_y:
            pt = sample(self.p, i, j, k)
            pb = sample(self.p, i, j - 1, k)
            if j - 1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_y[i, j, k] -= pt - pb
        for i, j, k in u_z:
            pc = sample(self.p, i, j, k)
            pa = sample(self.p, i, j, k - 1)
            if k - 1 < 0:
                pa = 0
            if j >= w_dim:
                pc = 0
            u_z[i, j, k] -= pc - pa

    def solve_pressure_MGPCG(self, verbose):
        self.init(self.u_div, -1)
        self.solve(max_iters=400, verbose=verbose, tol=1.e-12)
        self.get_result(self.p)

    @ti.kernel
    def set_uy_bottom(self):
        for i, j, k in self.u_y:
            if j == 0 and self.boundary_types[1, 0] == 2:
                self.u_y_bottom[i, j, k] = self.u_y[i, j, k]

    def Poisson(self, vel, verbose=False):
        """
        args:
            vel: torch tensor of shape (X, Y, Z, 3)
        returns:
            vel: torch tensor of shape (X, Y, Z, 3), projected
        """
        self.u.from_torch(vel)
        split_central_vector(self.u, self.u_x, self.u_y, self.u_z)
        self.set_uy_bottom()
        self.apply_bc(self.u_x, self.u_y, self.u_z)
        self.divergence(self.u_x, self.u_y, self.u_z)
        self.solve_pressure_MGPCG(verbose=verbose)
        self.subtract_grad_p(self.u_x, self.u_y, self.u_z)
        self.apply_bc(self.u_x, self.u_y, self.u_z)
        get_central_vector(self.u_x, self.u_y, self.u_z, self.u)
        vel = self.u.to_torch()
        return vel


############################################################################################################

def save_quiver_plot(u, v, res, save_path, scale=0.00000002):
    """
    Args:
        u: [H, W], vel along x (W)
        v: [H, W], vel along y (H)
        res: resolution of the plot along the longest axis; if None, let step = 1
        save_path:
    """
    import matplotlib.pyplot as plt
    import matplotlib
    H, W = u.shape
    y, x = np.mgrid[0:H, 0:W]
    axis_len = max(H, W)
    step = 1 if res is None else axis_len // res
    xq = [i[::step] for i in x[::step]]
    yq = [i[::step] for i in y[::step]]
    uq = [i[::step] for i in u[::step]]
    vq = [i[::step] for i in v[::step]]

    uv_norm = np.sqrt(np.array(uq) ** 2 + np.array(vq) ** 2).max()
    short_len = min(H, W)
    matplotlib.rcParams['font.size'] = 10 / short_len * axis_len
    fig, ax = plt.subplots(figsize=(10 / short_len * W, 10 / short_len * H))
    q = ax.quiver(xq, yq, uq, vq, pivot='tail', angles='uv', scale_units='xy', scale=scale / step)
    ax.invert_yaxis()
    plt.quiverkey(q, X=0.6, Y=1.05, U=uv_norm, label=f'Max arrow length = {uv_norm:.2g}', labelpos='E')
    plt.savefig(save_path)
    plt.close()
    return


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


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def batchify_query(inputs, query_function, batch_size=2 ** 22):
    """
    args:
        inputs: [..., input_dim]
    return:
        outputs: [..., output_dim]
    """
    input_dim = inputs.shape[-1]
    input_shape = inputs.shape
    inputs = inputs.view(-1, input_dim)  # flatten all but last dim
    N = inputs.shape[0]
    outputs = []
    for i in range(0, N, batch_size):
        output = query_function(inputs[i:i + batch_size])
        if isinstance(output, tuple):
            output = output[0]
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs.view(*input_shape[:-1], -1)  # unflatten


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


############################################################################################################

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


############################################################################################################


def merge_imgs(save_dir, framerate=30, prefix=''):
    os.system(
        'ffmpeg -hide_banner -loglevel error -y -i {0}/{1}%03d.png -vf palettegen {0}/palette.png'.format(save_dir,
                                                                                                          prefix))
    os.system(
        'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.gif'.format(
            framerate, save_dir, prefix))
    os.system(
        'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse -vcodec prores {1}/_{2}.mov'.format(
            framerate, save_dir, prefix))


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


############################################################################################################
from torch.func import vmap, jacrev

ti.init(arch=ti.cuda, device_memory_GB=12.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def batchify_rays(rays_flat, chunk=1024 * 64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify_get_ray_pts_velocity_and_derivitive(pts, chunk=1024 * 64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, pts.shape[0], chunk):
        ret = get_ray_pts_velocity_and_derivitives(pts[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def PDE_EQs(D_t, D_x, D_y, D_z, U, F, U_t=None, U_x=None, U_y=None, U_z=None, detach=False):
    eqs = []
    dts = [D_t]
    dxs = [D_x]
    dys = [D_y]
    dzs = [D_z]

    F = torch.cat([torch.zeros_like(F[:, :1]), F], dim=1) * 0  # (N,4)
    u, v, w = U.split(1, dim=-1)  # (N,1)
    F_t, F_x, F_y, F_z = F.split(1, dim=-1)  # (N,1)
    dfs = [F_t, F_x, F_y, F_z]

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim=-1)  # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim=-1)  # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim=-1)  # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim=-1)  # [d_z, u_z, v_z, w_z]
    else:
        dfs = [F_t]

    for i, (dt, dx, dy, dz, df) in enumerate(zip(dts, dxs, dys, dzs, dfs)):
        if i == 0:
            _e = dt + (u * dx + v * dy + w * dz) + df
        else:
            if detach:
                _e = dt + (u.detach() * dx + v.detach() * dy + w.detach() * dz) + df
            else:
                _e = dt + (u * dx + v * dy + w * dz) + df
        eqs += [_e]

    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [dxs[1] + dys[2] + dzs[3]]

    return eqs


def render(H, W, K, rays=None, c2w=None,
           near=0., far=1., time_step=None,
           **kwargs):
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
    all_ret = batchify_rays(rays, **kwargs)
    if 'vel_map' in all_ret:
        k_extract = ['vel_map']
    elif 'rgb_map' in all_ret:
        k_extract = ['rgb_map']
    else:
        k_extract = []
    if N_t == 1:
        for k in k_extract:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = [{k: all_ret[k] for k in all_ret if k not in k_extract}, ]
    return ret_list + ret_dict


def get_velocity_and_derivitives(pts,
                                 **kwargs):
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

    # Render and reshape
    all_ret = batchify_get_ray_pts_velocity_and_derivitive(pts, **kwargs)

    k_extract = ['raw_vel', 'raw_f'] if kwargs['no_vel_der'] else ['raw_vel', 'raw_f', '_u_x', '_u_y', '_u_z', '_u_t']
    ret_list = [all_ret[k] for k in k_extract]
    return ret_list


def raw2outputs(raw, z_vals, rays_d, learned_rgb=None, render_vel=False):
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
    dists = torch.cat([dists, torch.tensor([0.1]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0.

    alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]  # [N_rays, N_samples]
    if render_vel:
        mask = raw[..., -1] > 0.1
        N_samples = raw.shape[1]
        rgb_map = raw[:, int(N_samples / 3.5), :3] * mask[:, int(N_samples / 3.5), None]
    else:
        rgb = torch.ones(3) * (0.6 + torch.tanh(learned_rgb) * 0.4)
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / (torch.sum(weights, -1) + 1e-10)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)
    depth_map[acc_map < 1e-1] = 0.

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_query_fn,
                N_samples,
                retraw=False,
                network_query_fn_vel=None,
                perturb=0.,
                ret_derivative=True,
                render_vel=False,
                render_sim=False,
                render_grid=False,
                den_grid=None,
                color_grid=None,
                sim_step=0,
                dt=None,
                **kwargs):
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
    time_step = ray_batch[0, -1]
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
    pts = torch.cat([pts, time_step * torch.ones((pts.shape[0], pts.shape[1], 1))], -1)  # [..., 4]
    pts_flat = torch.reshape(pts, [-1, 4])
    bbox_mask = bbox_model.insideMask(pts_flat[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True  # in case zero rays are inside the bbox
    pts = pts_flat[bbox_mask]
    ret = {}

    pts.requires_grad = True
    model = kwargs['network_fn']
    embed_fn = kwargs['embed_fn']

    def g(x):
        return model(x)

    h = embed_fn(pts)
    raw_d = model(h)
    jac = vmap(jacrev(g))(h)
    jac_x = _get_minibatch_jacobian(h, pts)
    jac = jac @ jac_x

    ret = {'raw_d': raw_d, 'pts': pts}
    _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]
    ret['_d_x'] = _d_x
    ret['_d_y'] = _d_y
    ret['_d_z'] = _d_z
    ret['_d_t'] = _d_t
    out_dim = 1
    raw_flat = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
    raw_flat[bbox_mask] = raw_d
    raw = raw_flat.reshape(N_rays, N_samples, out_dim)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d,
                                                                 learned_rgb=kwargs['network_fn'].rgb)
    ret['rgb_map'] = rgb_map
    ret['raw_d'] = raw_d

    return ret


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)

        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def get_ray_pts_velocity_and_derivitives(
        pts,
        network_vel_fn,
        N_samples,
        **kwargs):
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
    if kwargs['no_vel_der']:
        vel_output, f_output = network_vel_fn(pts)
        ret = {}
        ret['raw_vel'] = vel_output
        ret['raw_f'] = f_output
        return ret

    def g(x):
        return model(x)[0]

    model = kwargs['network_fn']
    embed_fn = kwargs['embed_fn']
    h = embed_fn(pts)
    vel_output, f_output = model(h)
    ret = {}
    ret['raw_vel'] = vel_output
    ret['raw_f'] = f_output
    if not kwargs['no_vel_der']:
        jac = vmap(jacrev(g))(h)
        jac_x = _get_minibatch_jacobian(h, pts)
        jac = jac @ jac_x
        assert jac.shape == (pts.shape[0], 3, 4)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
        d = _u_x[:, 0] + _u_y[:, 1] + _u_z[:, 2]
        ret['raw_vel'] = vel_output
        ret['_u_x'] = _u_x
        ret['_u_y'] = _u_y
        ret['_u_z'] = _u_z
        ret['_u_t'] = _u_t

    return ret


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


def train():
    args_npz = np.load("args.npz", allow_pickle=True)
    from types import SimpleNamespace
    args = SimpleNamespace(**{
        key: value.item() if isinstance(value, np.ndarray) and value.size == 1 else
        value.tolist() if isinstance(value, np.ndarray) else
        value
        for key, value in args_npz.items()
    })

    rx, ry, rz, proj_y, use_project, y_start = args.sim_res_x, args.sim_res_y, args.sim_res_z, args.proj_y, args.use_project, args.y_start
    boundary_types = ti.Matrix([[1, 1], [2, 1], [1, 1]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
    project_solver = MGPCG_3(boundary_types=boundary_types, N=[rx, proj_y, rz], base_level=3)

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

    global bbox_model
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale)
    print('Loaded scalarflow', images_train_.shape, render_poses.shape, hwf, args.datadir)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # Create nerf model
    bds_dict = {
        'near': near,
        'far': far,
    }

    # Create nerf model
    ############################
    from src.encoder2 import HashEncoderNative
    embed_fn = HashEncoderNative(max_res=args.finest_resolution, min_res=args.base_resolution, device=device)
    input_ch = embed_fn.num_scales * 2  # default 2 params per scale
    embedding_params = list(embed_fn.parameters())

    model = mmodel.NeRFSmall(num_layers=2,
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
    optimizer = radam.RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args.lrate_den, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    start = 0
    basedir = args.basedir
    expname = args.expname

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'embed_fn': embed_fn,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    ############################

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Create nerf model
    ############################
    embed_fn_v = HashEncoderNative(max_res=args.finest_resolution_v, min_res=args.base_resolution_v, device=device)
    input_ch_v = embed_fn_v.num_scales * 2  # default 2 params per scale
    embedding_params_v = list(embed_fn_v.parameters())

    model_v = mmodel.NeRFSmallPotential(num_layers=args.vel_num_layers,
                                        hidden_dim=64,
                                        geo_feat_dim=15,
                                        num_layers_color=2,
                                        hidden_dim_color=16,
                                        input_ch=input_ch_v,
                                        use_f=args.use_f).to(device)
    grad_vars_vel = list(model_v.parameters())
    print(model_v)
    print('Total number of trainable parameters in model: {}'.format(
        sum([p.numel() for p in model_v.parameters() if p.requires_grad])))
    print('Total number of parameters in embedding: {}'.format(
        sum([p.numel() for p in embedding_params_v if p.requires_grad])))

    # network_query_fn = lambda x: model(embed_fn(x))
    def network_vel_fn(x):
        with torch.enable_grad():
            if not args.no_vel_der:
                h = embed_fn_v(x)
                v, f = model_v(h)
                return v, f, h
            else:
                v, f = model_v(embed_fn_v(x))
                return v, f

    # Create optimizer
    optimizer_vel = torch.optim.RAdam([
        {'params': grad_vars_vel, 'weight_decay': 1e-6},
        {'params': embedding_params_v, 'eps': 1e-15}
    ], lr=args.lrate, betas=(0.9, 0.99))
    grad_vars_vel += list(embedding_params_v)
    start_vel = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    render_kwargs_train_vel = {
        'network_vel_fn': network_vel_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,

        'network_fn': model_v,
        'embed_fn': embed_fn_v,
    }

    render_kwargs_test_vel = {k: render_kwargs_train_vel[k] for k in render_kwargs_train_vel}
    render_kwargs_test_vel['perturb'] = False

    ############################

    render_kwargs_train_vel.update(bds_dict)
    render_kwargs_test_vel.update(bds_dict)
    global_step = start
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

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
    images_train = torch.Tensor(images_train).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
    # images_train = images_train.reshape((images_train.shape[0], -1, 3))
    T, S, _ = images_train.shape
    rays = torch.Tensor(rays).to(device)
    ray_idxs = torch.randperm(rays.shape[0])

    loss_list = []
    psnr_list = []
    start = start + 1
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    flow_loss_meter, scale_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter()
    u_loss_meter, v_loss_meter, w_loss_meter, d_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    proj_loss_meter = AverageMeter()
    den2vel_loss_meter = AverageMeter()
    vel_loss_meter = AverageMeter()

    print('creating grid')
    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)],
                                indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    print('done')

    print('start training: from {} to {}'.format(start, args.N_iters))

    resample_rays = False
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
        frames_interp = frames_floor * (1 - time_idx_residual).unsqueeze(-1) + \
                        frames_ceil * time_idx_residual.unsqueeze(-1)  # [N_t, VHW, 3]
        time_step = time_idx / (T - 1) if T > 1 else torch.zeros_like(time_idx)
        points = frames_interp[:, batch_ray_idx]  # [N_t, B, 3]
        # points = torch.from_numpy(points).to(device)
        target_s = points.flatten(0, 1)  # [N_t*B, 3]

        if i_batch >= rays.shape[0]:
            print("Shuffle data after an epoch!")
            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = True

        #####  Core optimization loop  #####
        optimizer.zero_grad()
        optimizer_vel.zero_grad()

        extras = render(H, W, K, rays=batch_rays, time_step=time_step,
                        **render_kwargs_train)
        rgb = extras[0]
        extras = extras[1]

        pts = extras['pts']
        if args.no_vel_der:
            raw_vel, raw_f = get_velocity_and_derivitives(pts, no_vel_der=True, **render_kwargs_train_vel)
            _u_x, _u_y, _u_z, _u_t = None, None, None, None
        else:
            raw_vel, raw_f, _u_x, _u_y, _u_z, _u_t = get_velocity_and_derivitives(pts, no_vel_der=False,
                                                                                  **render_kwargs_train_vel)
        _d_t = extras['_d_t']
        _d_x = extras['_d_x']
        _d_y = extras['_d_y']
        _d_z = extras['_d_z']

        split_nse = PDE_EQs(
            _d_t, _d_x, _d_y, _d_z,
            raw_vel, raw_f, _u_t, _u_x, _u_y, _u_z, detach=args.detach_vel)
        nse_errors = [mean_squared_error(x, 0.0) for x in split_nse]
        if torch.stack(nse_errors).sum() > 10000:
            print(f'skip large loss {torch.stack(nse_errors).sum():.3g}, timestep={pts[0, 3]}')
            continue

        nseloss_fine = 0.0
        split_nse_wei = [args.flow_weight, args.vel_weight, args.vel_weight, args.vel_weight, args.d_weight] if not args.no_vel_der \
            else [args.flow_weight]

        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        loss_meter.update(img_loss.item())
        psnr_meter.update(psnr.item())

        # adhoc
        flow_loss_meter.update(split_nse_wei[0] * nse_errors[0].item())
        scale_meter.update(nse_errors[-1].item())
        norm_meter.update((split_nse_wei[-1] * nse_errors[-1]).item())
        if not args.no_vel_der:
            u_loss_meter.update((nse_errors[1]).item())
            v_loss_meter.update((nse_errors[2]).item())
            w_loss_meter.update((nse_errors[3]).item())
            d_loss_meter.update((nse_errors[4]).item())

        for ei, wi in zip(nse_errors, split_nse_wei):
            nseloss_fine = ei * wi + nseloss_fine

        if args.proj_weight > 0:
            # initialize density field
            coord_time_step = torch.ones_like(coord_3d_world[..., :1]) * time_step[0]
            coord_4d_world = torch.cat([coord_3d_world, coord_time_step], dim=-1)  # [X, Y, Z, 4]
            vel_world = batchify_query(coord_4d_world, render_kwargs_train_vel['network_vel_fn'])  # [X, Y, Z, 3]
            # y_start = args.y_start
            vel_world_supervised = vel_world.detach().clone()
            # vel_world_supervised[:, y_start:y_start + proj_y] = project_solver.Poisson(
            #     vel_world_supervised[:, y_start:y_start + proj_y])

            vel_world_supervised[..., 2] *= -1
            vel_world_supervised[:, y_start:y_start + proj_y] = project_solver.Poisson(
                vel_world_supervised[:, y_start:y_start + proj_y])
            vel_world_supervised[..., 2] *= -1

            proj_loss = img2mse(vel_world_supervised, vel_world)
        else:
            proj_loss = torch.zeros_like(img_loss)

        if args.d2v_weight > 0:
            raw_d = extras['raw_d']
            viz_dens_mask = raw_d.detach() > 0.1
            vel_norm = raw_vel.norm(dim=-1, keepdim=True)
            min_vel_mask = vel_norm.detach() < args.coef_den2vel * raw_d.detach()
            vel_reg_mask = min_vel_mask & viz_dens_mask
            min_vel_reg_map = (args.coef_den2vel * raw_d - vel_norm) * vel_reg_mask.float()
            min_vel_reg = min_vel_reg_map.pow(2).mean()
            # ipdb.set_trace()
        else:
            min_vel_reg = torch.zeros_like(img_loss)

        proj_loss_meter.update(proj_loss.item())
        den2vel_loss_meter.update(min_vel_reg.item())

        vel_loss = nseloss_fine + args.rec_weight * img_loss + args.proj_weight * proj_loss + args.d2v_weight * min_vel_reg
        vel_loss_meter.update(vel_loss.item())
        vel_loss.backward()

        if args.debug:
            print('vel loss', vel_loss.item())
            print('img loss', args.rec_weight * img_loss.item())
            print('testing gradients')
            grad_vel = render_kwargs_train_vel['network_fn'].sigma_net[0].weight.grad
            print('vel', grad_vel)
            if grad_vel is not None:
                print('vel', grad_vel.max(), grad_vel.min(), grad_vel.shape)
            grad_hashtable = render_kwargs_train_vel['embed_fn'].hash_table.grad
            print('hashtable', grad_hashtable)
            if grad_hashtable is not None:
                print('hashtable', grad_hashtable.max(), grad_hashtable.min(), grad_hashtable.shape)
            grad_density = render_kwargs_train['network_fn'].sigma_net[0].weight.grad
            print('density', grad_density)
            if grad_density is not None:
                print('density', grad_density.max(), grad_density.min(), grad_density.shape)
            grad_hashtable = render_kwargs_train['embed_fn'].hash_table.grad
            print('hashtable', grad_hashtable)
            if grad_hashtable is not None:
                print('hashtable', grad_hashtable.max(), grad_hashtable.min(), grad_hashtable.shape)

        optimizer_vel.step()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_vel.param_groups:
            param_group['lr'] = new_lrate

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Rec Loss:{loss_meter.avg:.2g} PSNR:{psnr_meter.avg:.4g} Flow Loss: {flow_loss_meter.avg:.2g}, "
                f"U loss: {u_loss_meter.avg:.2g}, V loss: {v_loss_meter.avg:.2g}, W loss: {w_loss_meter.avg:.2g},"
                f" d loss: {d_loss_meter.avg:.2g}, proj Loss:{proj_loss_meter.avg:.2g}, den2vel loss:{den2vel_loss_meter.avg:.2g}, Vel Loss: {vel_loss_meter.avg:.2g} ")
            loss_list.append(loss_meter.avg)
            psnr_list.append(psnr_meter.avg)
            loss_psnr = {
                "losses": loss_list,
                "psnr": psnr_list,
            }
            loss_meter.reset()
            psnr_meter.reset()
            flow_loss_meter.reset()
            scale_meter.reset()
            vel_loss_meter.reset()
            norm_meter.reset()
            u_loss_meter.reset()
            v_loss_meter.reset()
            w_loss_meter.reset()
            d_loss_meter.reset()

            with open(os.path.join(basedir, expname, "loss_vs_time.json"), "w") as fp:
                json.dump(loss_psnr, fp)
        if resample_rays:
            print("Sampling new rays!")
            if rays is not None:
                del rays
                torch.cuda.empty_cache()
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
            images_train = torch.Tensor(images_train).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
            rays = torch.Tensor(rays).to(device)

            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = False

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
