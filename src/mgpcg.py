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
