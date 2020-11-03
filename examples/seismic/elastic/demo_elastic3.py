from devito import *
from examples.seismic.source import WaveletSource, RickerSource, TimeAxis
from examples.seismic import plot_image
import numpy as np

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
import argparse

from devito.logger import info
from devito import TimeFunction, Function, Dimension, Eq, Inc
from devito import Operator, norm, configuration
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

from devito.types.basic import Scalar, Symbol # noqa
from mpl_toolkits.mplot3d import Axes3D # noqa


parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-tn", "--tn", default=40,
                    type=float, help="Simulation time in millisecond")
parser.add_argument("-bs", "--bsizes", default=(8, 8, 32, 32), type=int, nargs="+",
                    help="Block and tile sizes")
parser.add_argument("-plotting", "--plotting", default=0,
                    type=bool, help="Turn ON/OFF plotting")

args = parser.parse_args()

# --------------------------------------------------------------------------------------

# Define the model parameters
nx, ny, nz = args.shape
shape = (nx, ny, nz)  # Number of grid point (nx, ny, nz)
spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)
so = args.space_order
extent = (6000., 6000, 6000)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1]/(shape[1]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[2]/(shape[2]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, y, z))


class DGaussSource(WaveletSource):

    def wavelet(self, f0, t):
        a = 0.002
        return -2.*a*(t - 1/f0) * np.exp(-a * (t - 1/f0)**2)


# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., args.tn
dt = (10. / np.sqrt(2.)) / 6.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
src.coordinates.data[:] = [100 , 100 , 100]

# Now we create the velocity and pressure fields

v = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)

# Now let's try and create the staggered updates
t = grid.stepping_dim
time = grid.time_dim

# We need some initial conditions
V_p = 2.0
V_s = 1.0
density = 1.8

# The source injection term
src_xx = src.inject(field=tau.forward[0, 0], expr=src)
src_yy = src.inject(field=tau.forward[1, 1], expr=src)
src_zz = src.inject(field=tau.forward[2, 2], expr=src)

# Thorbecke's parameter notation
cp2 = V_p*V_p
cs2 = V_s*V_s
ro = 1/density

mu = cs2*density
l = (cp2*density - 2*mu)

# fdelmodc reference implementation
u_v = Eq(v.forward, v + dt*ro*div(tau))
u_t = Eq(tau.forward, tau + dt * l * diag(div(v.forward)) +
         dt * mu * (grad(v.forward) + grad(v.forward).T))

op = Operator([u_v] + [u_t]  + src_xx + src_yy + src_zz)
# op = Operator(src_xx + src_zz)
configuration['autotuning']='off'
op()
configuration['autotuning']='off'

# plot_image(v[0].data[0,:], cmap="seismic"); pause(1)
# plot_image(tau[0].data[0,:], cmap="seismic"); pause(1)
# plot_image(tau[0].data[0,int(nx/2),:,:], cmap="seismic"); pause(1)
# plot_image(tau[0,1].data[0,int(nx/2),:,:], cmap="seismic"); pause(1)

# import pdb; pdb.set_trace()
norm_v0 = norm(v[0])
norm_v1 = norm(v[1])
norm_v2 = norm(v[2])
print(norm_v0)
print(norm_v1)
print(norm_v2)

norm_t00 = norm(tau[0, 0])
norm_t11 = norm(tau[1, 1])
norm_t22 = norm(tau[2, 2])
print(norm_t00)
print(norm_t11)
print(norm_t22)

print("Let's inspect")
# import pdb; pdb.set_trace()

# f : perform source injection on an empty grid
# Inspection The source injection term
ftau = TensorTimeFunction(name='ftau', grid=grid, space_order=so, time_order=1)

src_fxx = src.inject(field=ftau.forward[0, 0], expr=src)
src_fyy = src.inject(field=ftau.forward[1, 1], expr=src)
src_fzz = src.inject(field=ftau.forward[2, 2], expr=src)

op_f = Operator(src_fxx + src_fyy + src_fzz)
op_f()

normf = norm(ftau[0])
normf = norm(ftau[1])

print("==========")
print(norm(ftau[0]))
print(norm(ftau[1]))
print("===========")

# Get the nonzero indices
nzinds = np.nonzero(ftau[0].data[0])  # nzinds is a tuple
assert len(nzinds) == len(shape)

source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.int32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.int32)
info("source_id data indexes start from 0 now !!!")

# source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(1, len(nzinds[0])+1))
source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(len(nzinds[0])))

source_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1
# plot3d(source_mask.data, model)

info("Number of unique affected points is: %d", len(nzinds[0])+1)

# Assert that first and last index are as expected
assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1], nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(ftau[0].data[0])))

info("At this point source_mask and source_id have been populated correctly")

nnz_shape = (grid.shape[0], grid.shape[1])  # Change only 3rd dim

nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x,y ), space_order=0, dtype=np.int32)


nnz_sp_source_mask.data[:, :] = source_mask.data[:, :, :].sum(2)
inds = np.where(source_mask.data == 1.)

maxz = len(np.unique(inds[-1]))
sparse_shape = (grid.shape[0], grid.shape[1], maxz)  # Change only 3rd dim

assert(len(nnz_sp_source_mask.dimensions) == (len(source_mask.dimensions)-1))

# Note:sparse_source_id is not needed as long as sparse info is kept in mask
# sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

id_dim = Dimension(name='id_dim')
b_dim = Dimension(name='b_dim')

# import pdb; pdb.set_trace()


v_sol = VectorTimeFunction(name='v_sol', grid=grid, space_order=so, time_order=1)
tau_sol = TensorTimeFunction(name='tau_sol', grid=grid, space_order=so, time_order=1)

save_src_fxx = TimeFunction(name='save_src_fxx', shape=(src.shape[0],
                            nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))
save_src_fyy = TimeFunction(name='save_src_fyy', shape=(src.shape[0],
                            nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))
save_src_fzz = TimeFunction(name='save_src_fzz', shape=(src.shape[0],
                            nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

src_fxx = src.inject(field=tau_sol.forward[0, 0], expr=src)
src_fyy = src.inject(field=tau_sol.forward[1, 1], expr=src)
src_fzz = src.inject(field=tau_sol.forward[2, 2], expr=src)


save_src_fxx_term = src.inject(field=save_src_fxx[src.dimensions[0], source_id], expr=src)
save_src_fyy_term = src.inject(field=save_src_fyy[src.dimensions[0], source_id], expr=src)
save_src_fzz_term = src.inject(field=save_src_fzz[src.dimensions[0], source_id], expr=src)

op1 = Operator(save_src_fxx_term + save_src_fyy_term + save_src_fzz_term)
op1()

sp_zi = Dimension(name='sp_zi')

sp_source_mask = Function(name='sp_source_mask', shape=(list(sparse_shape)), dimensions=(x, y, sp_zi), space_order=0, dtype=np.int32)

# Now holds IDs
sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[-1][:len(np.unique(inds[-1]))])

assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
assert(len(sp_source_mask.dimensions) == 3)

# import pdb; pdb.set_trace()

zind = Scalar(name='zind', dtype=np.int32)
xb_size = Scalar(name='xb_size', dtype=np.int32)
yb_size = Scalar(name='yb_size', dtype=np.int32)
x0_blk0_size = Scalar(name='x0_blk0_size', dtype=np.int32)
y0_blk0_size = Scalar(name='y0_blk0_size', dtype=np.int32)

block_sizes = Function(name='block_sizes', shape=(4, ), dimensions=(b_dim,), space_order=0, dtype=np.int32)
block_sizes.data[:] = args.bsizes

eqxb = Eq(xb_size, block_sizes[0])
eqyb = Eq(yb_size, block_sizes[1])
eqxb2 = Eq(x0_blk0_size, block_sizes[2])
eqyb2 = Eq(y0_blk0_size, block_sizes[3])

# ===============================================

# fdelmodc reference implementation
u_v_sol = Eq(v_sol.forward, v_sol + dt*ro*div(tau_sol))
u_t_sol = Eq(tau_sol.forward, tau_sol + dt * l * diag(div(v_sol.forward)) + dt * mu * (grad(v_sol.forward) + grad(v_sol.forward).T))
# op = Operator([u_v] + [u_t]  + src_xx + src_zz)


eq0 = Eq(sp_zi.symbolic_max, nnz_sp_source_mask[x, y] - 1, implicit_dims=(time, x, y))
# eq1 = Eq(zind, sp_source_mask[x, sp_zi], implicit_dims=(time, x, sp_zi))
eq1 = Eq(zind, sp_source_mask[x, y, sp_zi], implicit_dims=(time, x, y, sp_zi))

myexpr_fxx = source_mask[x, y, zind] * save_src_fxx[time, source_id[x, y, zind]]
myexpr_fyy = source_mask[x, y, zind] * save_src_fyy[time, source_id[x, y, zind]]
myexpr_fzz = source_mask[x, y, zind] * save_src_fzz[time, source_id[x, y, zind]]

# import pdb; pdb.set_trace()
eq_fxx = Inc(tau_sol[0].forward[t+1, x, y, zind], myexpr_fxx, implicit_dims=(time, x, y, sp_zi))
eq_fyy = Inc(tau_sol[4].forward[t+1, x, y, zind], myexpr_fyy, implicit_dims=(time, x, y, sp_zi))
eq_fzz = Inc(tau_sol[8].forward[t+1, x, y, zind], myexpr_fzz, implicit_dims=(time, x, y, sp_zi))

print("-----")
op2 = Operator([eqxb, eqyb, eqxb2, eqyb2, eq0, eq1, u_v_sol, u_t_sol, eq_fxx, eq_fyy, eq_fzz])
# print(op2.ccode)
print("===Temporal blocking==========")
op2()
print("===========")

configuration['jit-backdoor'] = False

# import pdb; pdb.set_trace()

print("Norm(f):", normf)

print("===========")
print("Norm(tau_sol0):", norm(tau_sol[0]))
print("Norm(tau_sol1):", norm(tau_sol[1]))
print("Norm(tau_sol2):", norm(tau_sol[2]))
print("Norm(tau_sol3):", norm(tau_sol[3]))
print("===========")

print("===========")
print("Norm(tau0):", norm(tau[0]))
print("Norm(tau1):", norm(tau[1]))
print("Norm(tau2):", norm(tau[2]))
print("Norm(tau3):", norm(tau[3]))
print("===========")

# save_src.data[0, source_id.data[14, 14, 11]]
# save_src.data[0 ,source_id.data[14, 14, sp_source_mask.data[14, 14, 0]]]

#plt.imshow(uref.data[2, int(nx/2) ,:, :]); pause(1)
#plt.imshow(usol.data[2, int(nx/2) ,:, :]); pause(1)


# assert np.isclose(normuref, normusol, atol=1e-06)
# import pdb; pdb.set_trace()

#import pyvista as pv

# cmap = plt.cm.get_cmap("viridis")
# Copy devito u data
#values = v_sol[1].data[0, :, :, :]

# vistagrid = pv.UniformGrid()
# vistagrid.dimensions = np.array(values.shape) + 1
# vistagrid.origin = (0, 0, 0)  # The bottom left corner of the data set
# vistagrid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
# vistagrid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!
# vistagrid.plot(show_edges=True)
# vistaslices = vistagrid.slice_orthogonal()
# vistaslices.plot(cmap=cmap)


# import pdb; pdb.set_trace()
# Uncomment to plot a slice of the field
# plt.imshow(usol.data[2, int(nx/2) ,:, :]); pause(1)

if args.plotting:
    plot_image(v[0].data[0,:,int(ny/2),:], cmap="seismic"); pause(1)
    plot_image(v_sol[0].data[0,:,int(ny/2),:], cmap="seismic"); pause(1)

    plot_image(v[1].data[0,:,:,int(nz/2)], cmap="seismic"); pause(1)
    plot_image(v_sol[1].data[0,:,:,int(nz/2)], cmap="seismic"); pause(1)

    plot_image(v[2].data[0,:,:,int(nz/2)], cmap="seismic"); pause(1)
    plot_image(v_sol[2].data[0,:,:,int(nz/2)], cmap="seismic"); pause(1)

    plot_image(tau[0,0].data[0, int(nx/2), :, :], cmap="seismic"); pause(1)
    plot_image(tau_sol[0,0].data[0, int(nx/2), :, :], cmap="seismic"); pause(1)

    plot_image(tau[1, 1].data[0, :, int(ny/2), :], cmap="seismic"); pause(1)
    plot_image(tau_sol[1, 1].data[0, :, int(ny/2), :], cmap="seismic"); pause(1)

    plot_image(tau[2, 2].data[0, :, :, int(nz/2)], cmap="seismic"); pause(1)
    plot_image(tau_sol[2, 2].data[0, :, :, int(nz/2)], cmap="seismic"); pause(1)


# plot_image(tau[0].data[0, :, :], cmap="seismic"); pause(1)
# plot_image(tau[0,1].data[0, :, :], cmap="seismic"); pause(1)
# plot_image(tau[0,1].data[0, :, :]); pause(1)
# plot_image(tau[0].data[0, :, :]); pause(1)


assert np.isclose(norm(tau[0]), norm(tau_sol[0]), atol=1e-06)
assert np.isclose(norm(tau[3]), norm(tau_sol[3]), atol=1e-06)
