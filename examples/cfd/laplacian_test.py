import numpy as np
from devito import Grid, Eq, solve, TimeFunction, Operator, configuration
from examples.cfd import plot_field, init_hat
from examples.seismic import RickerSource, TimeAxis


configuration['openmp'] = 1
configuration['dse'] = 'advanced'
configuration['dle'] = 'advanced'

# Some variable declarations
nx = 81
ny = 81
nz = 81
nt = 10
c = 1.
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
dz = 2. / (nz - 1)
print("dx %s, dy %s, dz %s" % (dx, dy, dz))
sigma = .2
dt = sigma * dx

grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=8)
init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)

pde = u.laplace

#op = Operator(Eq(u.forward, pde))

# print(op.ccode)

print('-----------------------------------------------')
# add sources

t0 = 0  # Simulation starts a t=0
tn = 1000.  # Simulation last 1 second (1000 ms)
dt = 0.01  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)


f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=1, time_range=time_range)

domain_size = (1000.0, 1000.0, 1000.0)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(domain_size) * .5
src.coordinates.data[0, -1] = 20.  # Depth is 20m
m = 1
# We can plot the time signature to see the wavelet
# src.show()
src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

op = Operator([Eq(u.forward, pde)] + src_term)



# print(op.ccode)


# test = op.ccode

op(time=3)
print(u.data[1,:,:,:])
print('--------------------------------------')
#print(type(op.ccode))