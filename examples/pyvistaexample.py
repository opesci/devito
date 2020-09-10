import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt
import numpy as np

mesh = examples.load_channels()
# define a categorical colormap
cmap = plt.cm.get_cmap("viridis", 4)


mesh.plot(cmap=cmap)


slices = mesh.slice_orthogonal()

import pdb; pdb.set_trace()
slices.plot(cmap=cmap)

slices = mesh.slice_orthogonal(x=20, y=20, z=30)

slices.plot(cmap=cmap)


