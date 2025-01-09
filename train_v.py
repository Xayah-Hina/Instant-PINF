import numpy as np
import torch
from tqdm import tqdm, trange

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

############################################################################################################
# from run_nerf_helpers import NeRFSmall, NeRFSmallPotential, save_quiver_plot, get_rays_np, get_rays, get_rays_np_continuous, to8b, batchify_query, sample_bilinear, img2mse, mse2psnr

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
            h = F.relu(h, inplace=True)

        sigma = h
        return sigma


class NeRFSmallPotential(torch.nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 use_f=False
                 ):
        super(NeRFSmallPotential, self).__init__()

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
                out_dim = hidden_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = torch.nn.ModuleList(sigma_net)
        self.out = torch.nn.Linear(hidden_dim, 3, bias=True)
        self.use_f = use_f
        if use_f:
            self.out_f = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.out_f2 = torch.nn.Linear(hidden_dim, 3, bias=True)

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, True)

        v = self.out(h)
        if self.use_f:
            f = self.out_f(h)
            f = F.relu(f, True)
            f = self.out_f2(f)
        else:
            f = v * 0
        return v, f


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


############################################################################################################
# from radam import RAdam

class RAdam(torch.optim.Optimizer):

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


def advect_SL_particle(particle_pos, vel_world_prev, coord_3d_sim, dt, RK=2, y_start=48, proj_y=128,
                       use_project=False, project_solver=None, bbox_model=None, **kwargs):
    """Advect a scalar quantity using a given velocity field.
    Args:
        particle_pos: [N, 3], in world coordinate domain
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        new_particle_pos: [N, 3], in simulation coordinate domain
    """
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_world[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start + proj_y]) if use_project else vel_world[:, y_start:y_start + proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        vel_world[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start + proj_y]) if use_project else vel_world[:, y_start:y_start + proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = F.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True).squeeze(0).permute(3, 2, 1, 0)  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    particle_pos_sampled = bbox_model.world2sim(particle_pos) * 2 - 1  # ranging [-1, 1]
    particle_vel_sim = F.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], particle_pos_sampled[None, None, None], align_corners=True).permute([0, 2, 3, 4, 1]).flatten(0, 3)  # [N, 3]
    particle_pos_new = particle_pos + dt * bbox_model.sim2world_rot(particle_vel_sim)  # [N, 3]
    return particle_pos_new


def advect_maccormack_particle(particle_pos, vel_world_prev, coord_3d_sim, dt, **kwargs):
    """
    Args:
        particle_pos: [N, 3], in world coordinate domain
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        particle_pos_new: [N, 3], in simulation coordinate domain
    """
    particle_pos_next = advect_SL_particle(particle_pos, vel_world_prev, coord_3d_sim, dt, **kwargs)
    particle_pos_back = advect_SL_particle(particle_pos_next, vel_world_prev, coord_3d_sim, -dt, **kwargs)
    particle_pos_new = particle_pos_next + (particle_pos - particle_pos_back) / 2
    return particle_pos_new


def get_particle_vel_der(particle_pos_3d_world, bbox_model, get_vel_der_fn, t):
    time_step = torch.ones_like(particle_pos_3d_world[..., :1]) * t
    particle_pos_4d_world = torch.cat([particle_pos_3d_world, time_step], dim=-1)  # [P, 4]
    particle_pos_4d_world.requires_grad_()
    with torch.enable_grad():
        _, _, _u_x, _u_y, _u_z, _u_t = get_vel_der_fn(particle_pos_4d_world)  # [P, 3], partial der of u,v,w
    jac = torch.stack([_u_x, _u_y, _u_z], dim=-1)  # [P, 3, 3]
    grad_u_world, grad_v_world, grad_w_world = jac[:, 0], jac[:, 1], jac[:, 2]  # [P, 3]
    return grad_u_world, grad_v_world, grad_w_world


def stretch_vortex_particles(particle_dir, grad_u, grad_v, grad_w, dt):
    stretch_term = torch.cat([(particle_dir * grad_u).sum(dim=-1, keepdim=True),
                              (particle_dir * grad_v).sum(dim=-1, keepdim=True),
                              (particle_dir * grad_w).sum(dim=-1, keepdim=True), ], dim=-1)  # [P, 3]
    particle_dir = particle_dir + stretch_term * dt
    particle_int = torch.norm(particle_dir, dim=-1, keepdim=True)
    particle_dir = particle_dir / (particle_int + 1e-8)
    return particle_dir, particle_int


def compute_curl(pts, get_vel_der_fn):
    """
    :param pts: [..., 4]
    :param get_vel_der_fn: function
    :return:
        curl: [..., 3]
    """
    pts_shape = pts.shape
    pts = pts.view(-1, pts_shape[-1])  # [N, 3]
    pts.requires_grad_()
    with torch.enable_grad():
        _, _, _u_x, _u_y, _u_z, _u_t = get_vel_der_fn(pts)  # [N, 3], partial der of u,v,w
    jac = torch.stack([_u_x, _u_y, _u_z], dim=-1)  # [N, 3, 3]
    curl = torch.stack([jac[:, 2, 1] - jac[:, 1, 2],
                        jac[:, 0, 2] - jac[:, 2, 0],
                        jac[:, 1, 0] - jac[:, 0, 1]], dim=-1)  # [N, 3]
    curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    return curl


def compute_curl_batch(pts, get_vel_der_fn, chunk=64 * 96 * 64):
    pts_shape = pts.shape
    pts = pts.view(-1, pts_shape[-1])  # [N, 3]
    N = pts.shape[0]
    curls = []
    for i in range(0, N, chunk):
        curl = compute_curl(pts[i:i + chunk], get_vel_der_fn)
        curls.append(curl)
    curl = torch.cat(curls, dim=0)  # [N, 3]
    curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    return curl


def generate_vort_trajectory_curl(time_steps, bbox_model, rx=128, ry=192, rz=128, get_vel_der_fn=None,
                                  P=100, N_sample=2 ** 10, den_net=None, **render_kwargs):
    print('Generating vortex trajectory using curl...')
    dt = time_steps[1] - time_steps[0]
    T = len(time_steps)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    # initialize density field
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[0]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]

    # place empty vortex particles
    all_init_pos = []
    all_init_dir = []
    all_init_int = []
    all_init_time = []

    for i in range(P):
        # sample 4d points
        timesteps = 0.25 + torch.rand(N_sample) * 0.65  # sample from t=0.25 to t=0.9
        sampled_3d_coord_x = 0.25 + torch.rand(N_sample) * 0.5  # [N]
        sampled_3d_coord_y = 0.25 + torch.rand(N_sample) * 0.5  # [N]
        sampled_3d_coord_z = 0.25 + torch.rand(N_sample) * 0.5  # [N]
        sampled_3d_coord = torch.stack([sampled_3d_coord_x, sampled_3d_coord_y, sampled_3d_coord_z], dim=-1)  # [N, 3]
        sampled_3d_coord_world = bbox_model.sim2world(sampled_3d_coord)  # [N, 3]
        sampled_4d_coord_world = torch.cat([sampled_3d_coord_world, timesteps[:, None]], dim=-1)  # [N, 4]

        # compute curl of sampled points
        density = den_net(sampled_4d_coord_world)  # [N, 1]
        density = density.squeeze(-1)  # [N]
        mask = density > 1
        curls = compute_curl_batch(sampled_4d_coord_world, get_vel_der_fn)  # [N, 3]
        curls = curls[mask]
        timesteps = timesteps[mask]
        sampled_3d_coord_world = sampled_3d_coord_world[mask]
        curls_norm = curls.norm(dim=-1)  # [N]
        print(i, 'max curl norm: ', curls_norm.max().item())

        # get points with highest curl norm
        max_idx = curls_norm.argmax()  # get points with highest curl norm
        init_pos = sampled_3d_coord_world[max_idx]  # [3]
        init_dir = curls[max_idx] / curls_norm[max_idx]  # [3]
        init_int = curls_norm[max_idx]  # [1]
        init_time = timesteps[max_idx]  # [1]
        all_init_pos.append(init_pos)
        all_init_dir.append(init_dir)
        all_init_int.append(init_int)
        all_init_time.append(init_time)

    all_init_pos = torch.stack(all_init_pos, dim=0)  # [P, 3]
    all_init_dir = torch.stack(all_init_dir, dim=0)  # [P, 3]
    all_init_int = torch.stack(all_init_int, dim=0)[:, None]  # [P, 1]
    all_init_time = torch.stack(all_init_time, dim=0)[:, None]  # [P, 1]

    # initialize vortex particle position, direction, and when it spawns
    particle_start_timestep = all_init_time  # [P, 1]
    particle_start_timestep = torch.floor(particle_start_timestep * T).expand(-1, T)  # [P, T]
    particle_time_mask = torch.arange(T).unsqueeze(0).expand(P, -1) >= particle_start_timestep  # [P, T]
    particle_time_coef = particle_time_mask.float()  # [P, T]
    for time_coef in particle_time_coef:
        n = 20
        first_idx = time_coef.nonzero()[0]
        try:
            time_coef[first_idx:first_idx + n] = torch.linspace(0, 1, n)
        except:
            time_coef[first_idx:] = torch.linspace(0, 1, T - first_idx.item())
    particle_pos_world = all_init_pos  # [P, 3]
    particle_dir_world = all_init_dir  # [P, 3]
    particle_int_multiplier = torch.ones_like(all_init_int)  # [P, 1]
    particle_int = all_init_int.clone()  # [P, 1]

    all_pos = []
    all_dir = []
    all_int = []

    for i in range(T):
        # update simulation den and source den
        if i > 0:
            coord_4d_world[..., 3] = time_steps[i - 1]  # sample velocity at previous moment
            vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]

            # advect vortex particles
            mask_to_evolve = particle_time_mask[:, i]
            print('particles to evolve: ', mask_to_evolve.sum().item(), '/', P)
            if any(mask_to_evolve):
                particle_pos_world[mask_to_evolve] = advect_maccormack_particle(particle_pos_world[mask_to_evolve], vel, coord_3d_sim, dt, bbox_model=bbox_model, **render_kwargs)

                # stretch vortex particles
                grad_u, grad_v, grad_w = get_particle_vel_der(particle_pos_world[mask_to_evolve], bbox_model, get_vel_der_fn, time_steps[i - 1])
                particle_dir_world[mask_to_evolve], particle_int_multiplier[mask_to_evolve] = stretch_vortex_particles(particle_dir_world[mask_to_evolve], grad_u, grad_v, grad_w, dt)
                particle_int[mask_to_evolve] = particle_int[mask_to_evolve] * particle_int_multiplier[mask_to_evolve]
                particle_int[particle_int > all_init_int] = all_init_int[particle_int > all_init_int]

        all_pos.append(particle_pos_world.clone())
        all_dir.append(particle_dir_world.clone())
        all_int.append(particle_int.clone())
    particle_pos_world = torch.stack(all_pos, dim=0).permute(1, 0, 2)  # [P, T, 3]
    particle_dir_world = torch.stack(all_dir, dim=0).permute(1, 0, 2)  # [P, T, 3]
    particle_intensity = torch.stack(all_int, dim=0).permute(1, 0, 2)  # [P, T, 1]
    radius = 0.03 * torch.ones(P, 1)[:, None].expand(-1, T, -1)  # [P, T, 1]
    vort_particles = {'particle_time_mask': particle_time_mask,
                      'particle_pos_world': particle_pos_world,
                      'particle_dir_world': particle_dir_world,
                      'particle_intensity': particle_intensity,
                      'particle_time_coef': particle_time_coef,
                      'radius': radius}
    return vort_particles


############################################################################################################
# from load_scalarflow import load_pinf_frame_data
import cv2
import os
import imageio.v2 as imageio
import json

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius, rotZ=True, wx=0.0, wy=0.0, wz=0.0):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # wx,wy,wz, additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    ct = torch.Tensor([
        [1, 0, 0, wx],
        [0, 1, 0, wy],
        [0, 0, 1, wz],
        [0, 0, 0, 1]]).float()
    c2w = ct @ c2w

    return c2w


def load_pinf_frame_data(basedir, half_res=False, split='train'):
    # frame data
    all_imgs = []
    all_poses = []

    with open(os.path.join(basedir, 'info.json'), 'r') as fp:
        # read render settings
        meta = json.load(fp)
        near = float(meta['near'])
        far = float(meta['far'])
        radius = (near + far) * 0.5
        phi = float(meta['phi'])
        rotZ = (meta['rot'] == 'Z')
        r_center = np.float32(meta['render_center'])

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]],
                              axis=1)  # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'], [3])

        # read video frames
        # all videos should be synchronized, having the same frame_rate and frame_num

        video_list = meta[split + '_videos'] if (split + '_videos') in meta else meta['train_videos'][0:1]

        for video_id, train_video in enumerate(video_list):
            imgs = []

            f_name = os.path.join(basedir, train_video['file_name'])
            reader = imageio.get_reader(f_name, "ffmpeg")
            for frame_i in range(train_video['frame_num']):
                reader.set_image_index(frame_i)
                frame = reader.get_next_data()

                H, W = frame.shape[:2]
                camera_angle_x = float(train_video['camera_angle_x'])
                Focal = .5 * W / np.tan(.5 * camera_angle_x)
                imgs.append(frame)

            reader.close()
            imgs = (np.float32(imgs) / 255.)

            if half_res:
                H = H // 2
                W = W // 2
                Focal = Focal / 2.

                imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
                for i, img in enumerate(imgs):
                    imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                imgs = imgs_half_res

            all_imgs.append(imgs)
            all_poses.append(np.array(
                train_video['transform_matrix_list'][frame_i]
                if 'transform_matrix_list' in train_video else train_video['transform_matrix']
            ).astype(np.float32))

    imgs = np.stack(all_imgs, 0)  # [V, T, H, W, 3]
    imgs = np.transpose(imgs, [1, 0, 2, 3, 4])  # [T, V, H, W, 3]
    poses = np.stack(all_poses, 0)  # [V, 4, 4]
    hwf = np.float32([H, W, Focal])

    # set render settings:
    sp_n = 120  # an even number!
    sp_poses = [
        pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2])
        for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
    ]
    render_poses = torch.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
    render_timesteps = np.arange(sp_n) / (sp_n - 1)

    return imgs, poses, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far


############################################################################################################
from skimage.metrics import structural_similarity


def advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, RK=2, y_start=48, proj_y=128,
              use_project=False, project_solver=None, bbox_model=None, **kwargs):
    """Advect a scalar quantity using a given velocity field.
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        advected_quantity: [X, Y, Z, 1]
        vel_world: [X, Y, Z, 3]
    """
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_world[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start + proj_y]) if use_project else vel_world[:, y_start:y_start + proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        vel_world[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start + proj_y]) if use_project else vel_world[:, y_start:y_start + proj_y]
        # breakpoint()
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = F.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True, padding_mode='zeros').squeeze(0).permute(3, 2, 1, 0)  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    backtrace_coord = coord_3d_sim - dt * vel_sim  # [X, Y, Z, 3]
    backtrace_coord_sampled = backtrace_coord * 2 - 1  # ranging [-1, 1]
    q_grid = q_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, C, Z, Y, X] i.e., [N, C, D, H, W]
    q_backtraced = F.grid_sample(q_grid, backtrace_coord_sampled.permute(2, 1, 0, 3)[None, ...], align_corners=True, padding_mode='zeros')  # [N, C, D, H, W]
    q_backtraced = q_backtraced.squeeze(0).permute([3, 2, 1, 0])  # [X, Y, Z, C]
    return q_backtraced, vel_world


def advect_maccormack(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs):
    """
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        advected_quantity: [X, Y, Z, C]
        vel_world: [X, Y, Z, 3]
    """
    q_grid_next, _ = advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs)
    q_grid_back, vel_world = advect_SL(q_grid_next, vel_world_prev, coord_3d_sim, -dt, **kwargs)
    q_advected = q_grid_next + (q_grid - q_grid_back) / 2
    C = q_advected.shape[-1]
    for i in range(C):
        q_max, q_min = q_grid[..., i].max(), q_grid[..., i].min()
        q_advected[..., i] = q_advected[..., i].clamp_(q_min, q_max)
    return q_advected, vel_world


from lpips import LPIPS


def run_advect_den(render_poses, hwf, K, time_steps, savedir, gt_imgs, bbox_model, rx=128, ry=192, rz=128,
                   save_fields=False, save_den=False, vort_particles=None, render=None, get_vel_der_fn=None, **render_kwargs):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(chunk=512 * 16)
    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    # initialize density field
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[0]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]
    den = batchify_query(coord_4d_world, render_kwargs['network_query_fn'])  # [X, Y, Z, 1]
    den_ori = den
    vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]
    vel_saved = vel
    bbox_mask = bbox_model.insideMask(coord_3d_world[..., :3].reshape(-1, 3), to_float=False)
    bbox_mask = bbox_mask.reshape(rx, ry, rz)

    source_height = 0.25
    y_start = int(source_height * ry)
    print('y_start: {}'.format(y_start))
    render_kwargs.update(y_start=y_start)
    for i, c2w in enumerate(tqdm(render_poses)):
        # update simulation den and source den
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        if i > 0:
            coord_4d_world[..., 3] = time_steps[i - 1]  # sample velocity at previous moment

            vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]
            vel_saved = vel
            # advect vortex particles
            if vort_particles is not None:
                confinement_field = vort_particles(coord_3d_world, i)
                print('Vortex energy over velocity: {:.2f}%'.format(torch.norm(confinement_field, dim=-1).pow(2).sum() / torch.norm(vel, dim=-1).pow(2).sum() * 100))
            else:
                confinement_field = torch.zeros_like(vel)

            vel_confined = vel + confinement_field
            den, vel = advect_maccormack(den, vel_confined, coord_3d_sim, dt, bbox_model=bbox_model, **render_kwargs)
            den_ori = batchify_query(coord_4d_world, render_kwargs['network_query_fn'])  # [X, Y, Z, 1]
            # zero grad for coord_4d_world
            # coord_4d_world.grad = None
            # coord_4d_world = coord_4d_world.detach()

            coord_4d_world[..., 3] = time_steps[i]  # source density at current moment
            den[~mask_to_sim] = batchify_query(coord_4d_world[~mask_to_sim], render_kwargs['network_query_fn'])
            den[~bbox_mask] *= 0.0

        if save_fields:
            # save_fields_to_vti(vel.permute(2, 1, 0, 3).detach().cpu().numpy(),
            #                    den.permute(2, 1, 0, 3).detach().cpu().numpy(),
            #                    os.path.join(savedir, 'fields_{:03d}.vti'.format(i)))
            np.save(os.path.join(savedir, 'den_{:03d}.npy'.format(i)), den.permute(2, 1, 0, 3).detach().cpu().numpy())
            np.save(os.path.join(savedir, 'den_ori_{:03d}.npy'.format(i)), den_ori.permute(2, 1, 0, 3).detach().cpu().numpy())
            np.save(os.path.join(savedir, 'vel_{:03d}.npy'.format(i)), vel_saved.permute(2, 1, 0, 3).detach().cpu().numpy())
        if save_den:
            # save_vdb(den[..., 0].detach().cpu().numpy(),
            #          os.path.join(savedir, 'den_{:03d}.vdb'.format(i)))
            # save npy files
            np.save(os.path.join(savedir, 'den_{:03d}.npy'.format(i)), den[..., 0].detach().cpu().numpy())
        rgb, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_grid=True, den_grid=den,
                        **render_kwargs)
        rgb8 = to8b(rgb.detach().cpu().numpy())
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
        imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(i)), rgb8)
        imageio.imsave(os.path.join(savedir, 'gt_{:03d}.png'.format(i)), gt_img8)
    merge_imgs(savedir, prefix='rgb_')
    merge_imgs(savedir, prefix='gt_')

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"Avg PSNR over full simulation: ", avg_psnr)
        avg_ssim = sum(ssims) / len(ssims)
        print(f"Avg SSIM over full simulation: ", avg_ssim)
        avg_lpips = sum(lpipss) / len(lpipss)
        print(f"Avg LPIPS over full simulation: ", avg_lpips)
        with open(os.path.join(savedir, "psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)), "w") as fp:
            json.dump(psnrs, fp)


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
import torch.nn.functional as F
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


def render_path(render_poses, hwf, K, gt_imgs=None, savedir=None, time_steps=None, vel_scale=0.01, sim_step=5, **render_kwargs):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(dt=dt)
    render_kwargs.update(chunk=512 * 16)
    psnrs = []

    for i, c2w in enumerate(tqdm(render_poses)):
        vel_map, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_vel=True, **render_kwargs)
        vel_map = vel_map.cpu().numpy()  # [H, W, 2]
        # finite difference has issues with boundary because those are not seen during training. Remove those.
        vel_map[0], vel_map[-1], vel_map[:, 0], vel_map[:, -1] = 0, 0, 0, 0

        rgb, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_sim=True, sim_step=sim_step, **render_kwargs)
        rgb8 = to8b(rgb.cpu().numpy())
        if gt_imgs is not None:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            print(f'PSNR: {p:.4g}')
            psnrs.append(p)

        if savedir is not None:
            save_quiver_plot(vel_map[..., 0], vel_map[..., 1], 64, os.path.join(savedir, 'vel_{:03d}.png'.format(i)),
                             scale=vel_scale)
            imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(i)), rgb8)

    if savedir is not None:
        merge_imgs(savedir, prefix='vel_')
        merge_imgs(savedir, prefix='rgb_')

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"Avg PSNR over {sim_step}-step simulation: ", avg_psnr)
        with open(os.path.join(savedir, "{}step_psnrs_avg{:0.2f}.json".format(sim_step, avg_psnr)), "w") as fp:
            json.dump(psnrs, fp)

    return


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    from src.encoder import HashEncoderHyFluid
    # embed_fn, input_ch = get_encoder('hashgrid', input_dim=4, num_levels=args.num_levels, base_resolution=args.base_resolution,
    #                                  finest_resolution=args.finest_resolution, log2_hashmap_size=args.log2_hashmap_size,)
    max_res = np.array([args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
    min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])

    embed_fn = HashEncoderHyFluid(max_res=max_res, min_res=min_res, num_scales=args.num_levels,
                                  max_params=2 ** args.log2_hashmap_size)
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
    ], lr=args.lrate_den, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'embed_fn': embed_fn,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def create_vel_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    from src.encoder import HashEncoderHyFluid
    max_res = np.array([args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v_t])
    min_res = np.array([args.base_resolution_v, args.base_resolution_v, args.base_resolution_v, args.base_resolution_v_t])

    embed_fn = HashEncoderHyFluid(max_res=max_res, min_res=min_res, num_scales=args.num_levels,
                                  max_params=2 ** args.log2_hashmap_size)
    input_ch = embed_fn.num_scales * 2  # default 2 params per scale
    embedding_params = list(embed_fn.parameters())

    model = NeRFSmallPotential(num_layers=args.vel_num_layers,
                               hidden_dim=64,
                               geo_feat_dim=15,
                               num_layers_color=2,
                               hidden_dim_color=16,
                               input_ch=input_ch,
                               use_f=args.use_f).to(device)
    grad_vars = list(model.parameters())
    print(model)
    print('Total number of trainable parameters in model: {}'.format(
        sum([p.numel() for p in model.parameters() if p.requires_grad])))
    print('Total number of parameters in embedding: {}'.format(
        sum([p.numel() for p in embedding_params if p.requires_grad])))

    # network_query_fn = lambda x: model(embed_fn(x))
    def network_vel_fn(x):
        with torch.enable_grad():
            if not args.no_vel_der:
                h = embed_fn(x)
                v, f = model(h)
                return v, f, h
            else:
                v, f = model(embed_fn(x))
                return v, f

    # Create optimizer
    optimizer = torch.optim.RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args.lrate, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_v_path is not None and args.ft_v_path != 'None':
        ckpts = [args.ft_v_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        print(ckpt['vel_network_fn_state_dict'].keys())
        # update model
        model_dict = model.state_dict()
        pretrained_dict = ckpt['vel_network_fn_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Updated parameters:{}/{}".format(len(pretrained_dict), len(model_dict)))
        # model.load_state_dict(ckpt['vel_network_fn_state_dict'])
        embed_fn.load_state_dict(ckpt['vel_embed_fn_state_dict'])

        optimizer.load_state_dict(ckpt['vel_optimizer_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_vel_fn': network_vel_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,

        'network_fn': model,
        'embed_fn': embed_fn,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


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
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

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
    if render_vel:
        out_dim = 3
        raw_flat_vel = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_vel[bbox_mask] = network_query_fn_vel(pts)[0]  # raw_vel
        raw_vel = raw_flat_vel.reshape(N_rays, N_samples, out_dim)
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_den[bbox_mask] = network_query_fn(pts)  # raw_den
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
        raw = torch.cat([raw_vel, raw_den], -1)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, render_vel=render_vel)
        vel_map = rgb_map[..., :2]
        ret['vel_map'] = vel_map
    elif render_sim:
        assert dt is not None and dt > 0, 'dt must be specified a positive number for sim_onestep'
        for i in range(sim_step):
            if pts[0, 3] - dt < 0:
                break

            MacCormack = False  # It marginally (but consistently) improves, but slower. Don't use it until final results.
            if not MacCormack:  # semi-lag for backtracing
                raw_vel = network_query_fn_vel(pts)[0]  # raw_vel
                pts[..., :3] = pts[..., :3] - dt * raw_vel
                pts[..., 3] = pts[..., 3] - dt
            else:  # MacCormack advection
                raw_vel = network_query_fn_vel(pts)[0]
                one_step_back_pts = pts.clone()
                one_step_back_pts[..., :3] = pts[..., :3] - dt * raw_vel
                one_step_back_pts[..., 3] = pts[..., 3] - dt
                returning_vel = network_query_fn_vel(one_step_back_pts)[0]
                returning_pts = one_step_back_pts.clone()
                returning_pts[..., :3] = one_step_back_pts[..., :3] + dt * returning_vel
                returning_pts[..., 3] = one_step_back_pts[..., 3] + dt
                pts_maccorck = one_step_back_pts.clone()
                pts_maccorck[..., :3] = pts_maccorck[..., :3] + (pts[..., :3] - returning_pts[..., :3]) / 2
                pts = pts_maccorck

        # query density
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_den[bbox_mask] = network_query_fn(pts)  # raw_den
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_den, z_vals, rays_d, learned_rgb=kwargs['network_fn'].rgb)
        ret['rgb_map'] = rgb_map
    elif render_grid:  # render from a voxel grid
        assert den_grid is not None, 'den_grid must be specified for render_grid.'
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)

        pts_world = pts[..., :3]
        pts_sim = bbox_model.world2sim(pts_world)
        pts_sample = pts_sim * 2 - 1  # ranging [-1, 1]
        den_grid = den_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, 1, Z, Y, X] i.e., [N, 1, D, H, W]
        den_sampled = F.grid_sample(den_grid, pts_sample[None, ..., None, None, :], align_corners=True)

        raw_flat_den[bbox_mask] = den_sampled.reshape(-1, 1)
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)

        if color_grid is not None:
            raw_flat_rgb = torch.zeros([N_rays, N_samples, 3]).reshape(-1, 3)
            color_grid = color_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, 1, Z, Y, X] i.e., [N, 3, D, H, W]
            color_sampled = F.grid_sample(color_grid, pts_sample[None, ..., None, None, :], align_corners=True)
            raw_flat_rgb[bbox_mask] = color_sampled.reshape(-1, 1)
            raw_rgb = raw_flat_rgb.reshape(N_rays, N_samples, 3)
        else:
            raw_rgb = None

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_den, z_vals, rays_d, learned_rgb=kwargs['network_fn'].rgb if color_grid is None else raw_rgb)
        ret['rgb_map'] = rgb_map
    else:  # get density gradient for flow loss
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

    # Load data
    images_train_, poses_train, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far = \
        load_pinf_frame_data(args.datadir, args.half_res, split='train')
    images_test, poses_test, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far = \
        load_pinf_frame_data(args.datadir, args.half_res, split='test')
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
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train_vel, render_kwargs_test_vel, start_vel, grad_vars_vel, optimizer_vel = create_vel_nerf(args)
    render_kwargs_train_vel.update(bds_dict)
    render_kwargs_test_vel.update(bds_dict)
    global_step = start
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            test_view_pose = torch.tensor(poses_test[0])
            N_timesteps = images_test.shape[0]
            test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
            test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
            print(test_view_poses.shape)
            render_kwargs_test.update(network_query_fn_vel=render_kwargs_test_vel['network_vel_fn'])
            render_path(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir, vel_scale=args.vel_scale,
                        gt_imgs=images_test, save_fields=args.save_fields, **render_kwargs_test)
            return
    if args.run_advect_den:
        print('Run advect density.')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'run_advect_den_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            test_view_pose = torch.tensor(poses_test[0])
            N_timesteps = images_test.shape[0]
            test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
            test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
            render_kwargs_test.update(network_query_fn_vel=render_kwargs_test_vel['network_vel_fn'])
            get_vel_der_fn = lambda pts: get_velocity_and_derivitives(pts, no_vel_der=False, **render_kwargs_test_vel)

            if args.generate_vort_particles:
                vort_particles = generate_vort_trajectory_curl(time_steps=test_timesteps,
                                                               bbox_model=bbox_model, rx=rx, ry=ry, rz=rz,
                                                               get_vel_der_fn=get_vel_der_fn,
                                                               **render_kwargs_test)
            else:
                vort_particles = None
            run_advect_den(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                           gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                           proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                           save_den=args.save_den, get_vel_der_fn=get_vel_der_fn, vort_particles=vort_particles,
                           save_fields=args.save_fields, **render_kwargs_test)
            run_advect_den(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                           gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                           proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                           **render_kwargs_test)
            return

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
        ################################
        # Rest is logging
        if i % args.i_weights == 0:
            os.makedirs(os.path.join(basedir, expname, 'den'), exist_ok=True)
            path = os.path.join(basedir, expname, 'den', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'vel_network_fn_state_dict': render_kwargs_train_vel['network_fn'].state_dict(),
                'vel_embed_fn_state_dict': render_kwargs_train_vel['embed_fn'].state_dict(),
                'vel_optimizer_state_dict': optimizer_vel.state_dict(),
            }, path)

            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            if i % (args.i_video) == 0:
                print('Run advect density.')
                with torch.no_grad():
                    testsavedir = os.path.join(basedir, expname, 'run_advect_den_{:06d}'.format(i))
                    os.makedirs(testsavedir, exist_ok=True)
                    test_view_pose = torch.tensor(poses_test[0])
                    N_timesteps = images_test.shape[0]
                    test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
                    test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
                    render_kwargs_test.update(network_query_fn_vel=render_kwargs_test_vel['network_vel_fn'])
                    run_advect_den(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                                   gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                                   proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                                   **render_kwargs_test)
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
    import ipdb

    train()
