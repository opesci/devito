from devito import *
from examples.seismic.source import WaveletSource, RickerSource, TimeAxis
from examples.seismic import plot_image
import numpy as np

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
import argparse

from devito.logger import info
from devito import TimeFunction, Function, Dimension, Eq, Inc
from devito import Operator, norm
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
extent = (600., 600, 600)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1]/(shape[1]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[2]/(shape[2]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, y, z))
