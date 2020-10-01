# coding: utf-8
from devito import TimeFunction, warning
from devito.tools import memoized_meth
from examples.seismic.tti.operators import ForwardOperator, AdjointOperator
from examples.seismic.tti.operators import particle_velocity_fields
from examples.seismic import PointSource, Receiver
from devito import norm, Operator, Function, Dimension, Eq, Inc
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from devito.types.basic import Scalar
from matplotlib.pyplot import pause # noqa
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size


class AnisotropicWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.

    Notes
    -----
    space_order must be even and it is recommended to be a multiple of 4
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        if space_order % 2 != 0:
            raise ValueError("space_order must be even but got %s"
                             % space_order)

        if space_order % 4 != 0:
            warning("It is recommended for space_order to be a multiple of 4" +
                    "but got %s" % space_order)

        self.space_order = space_order

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, kernel='centered', save=False, tteqs=(), **kwargs):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order,
                               kernel=kernel, tteqs=tteqs, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, v=None, vp=None,
                epsilon=None, delta=None, theta=None, phi=None,
                save=False, kernel='centered', **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        u : TimeFunction, optional
            The computed wavefield first component.
        v : TimeFunction, optional
            The computed wavefield second component.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.
        kernel : str, optional
            Type of discretization, centered or shifted.

        Returns
        -------
        Receiver, wavefield and performance summary.
        """
        if kernel == 'staggered':
            time_order = 1
            dims = self.model.space_dimensions
            stagg_u = (-dims[-1])
            stagg_v = (-dims[0], -dims[1]) if self.model.grid.dim == 3 else (-dims[0])
        else:
            time_order = 2
            stagg_u = stagg_v = None
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)
        # Create the forward wavefield if not provided

        if u is None:
            u = TimeFunction(name='u', grid=self.model.grid, staggered=stagg_u,
                             save=self.geometry.nt if save else None,
                             time_order=time_order,
                             space_order=self.space_order)
        # Create the forward wavefield if not provided
        if v is None:
            v = TimeFunction(name='v', grid=self.model.grid, staggered=stagg_v,
                             save=self.geometry.nt if save else None,
                             time_order=time_order,
                             space_order=self.space_order)

        print("Initial Norm u", norm(u))
        print("Initial Norm v", norm(v))

        if kernel == 'staggered':
            vx, vz, vy = particle_velocity_fields(self.model, self.space_order)
            kwargs["vx"] = vx
            kwargs["vz"] = vz
            if vy is not None:
                kwargs["vy"] = vy

        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(self.model.physical_params(
            vp=vp, epsilon=epsilon, delta=delta, theta=theta, phi=phi)
        )
        if self.model.dim < 3:
            kwargs.pop('phi', None)
        # Execute operator and return wavefield and receiver data

        op = self.op_fwd(kernel, save)
        print(kwargs)
        summary = op.apply(src=src, u=u, v=v,
                           dt=kwargs.pop('dt', self.dt), **kwargs)

        regnormu = norm(u)
        regnormv = norm(v)
        print("Norm u:", regnormu)
        print("Norm v:", regnormv)

        if 0:
            cmap = plt.cm.get_cmap("viridis")
            values = u.data[0, :, :, :]
            vistagrid = pv.UniformGrid()
            vistagrid.dimensions = np.array(values.shape) + 1
            vistagrid.spacing = (1, 1, 1)
            vistagrid.origin = (0, 0, 0)  # The bottom left corner of the data set
            vistagrid.cell_arrays["values"] = values.flatten(order="F")
            vistaslices = vistagrid.slice_orthogonal()
            vistagrid.plot(show_edges=True)
            vistaslices.plot(cmap=cmap)

        print("=========================================") 

        s_u = TimeFunction(name='s_u', grid=self.model.grid, space_order=self.space_order, time_order=1)
        s_v = TimeFunction(name='s_v', grid=self.model.grid, space_order=self.space_order, time_order=1)

        src_u = src.inject(field=s_u.forward, expr=src* self.model.grid.time_dim.spacing**2 / self.model.m)
        src_v = src.inject(field=s_v.forward, expr=src * self.model.grid.time_dim.spacing**2 / self.model.m)

        op_f = Operator([src_u, src_v])
        op_f.apply(src=src, dt=kwargs.pop('dt', self.dt))

        print("Norm s_u", norm(s_u))
        print("Norm s_v", norm(s_v))


        # Get the nonzero indices
        nzinds = np.nonzero(s_u.data[0])  # nzinds is a tuple
        assert len(nzinds) == len(self.model.grid.shape)
        shape = self.model.grid.shape
        x, y, z = self.model.grid.dimensions
        time = self.model.grid.time_dim
        t = self.model.grid.stepping_dim

        source_mask = Function(name='source_mask', shape=self.model.grid.shape, dimensions=(x, y, z), space_order=0, dtype=np.int32)
        source_id = Function(name='source_id', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.int32)
        print("source_id data indexes start from 0 now !!!")

        # source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(1, len(nzinds[0])+1))
        source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(len(nzinds[0])))

        source_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1
        # plot3d(source_mask.data, model)
        # import pdb; pdb.set_trace()

        print("Number of unique affected points is: %d", len(nzinds[0]))

        # Assert that first and last index are as expected
        assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
        assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
        assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1], nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

        assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
        assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(s_u.data[0])))

        print("-At this point source_mask and source_id have been popoulated correctly-")

        nnz_shape = (self.model.grid.shape[0], self.model.grid.shape[1])

        nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x,y ), space_order=0, dtype=np.int32)

        nnz_sp_source_mask.data[:, :] = source_mask.data[:, :, :].sum(2)
        inds = np.where(source_mask.data == 1.)
        print("Grid - source positions:", inds)
        maxz = len(np.unique(inds[-1]))
        # Change only 3rd dim
        sparse_shape = (self.model.grid.shape[0], self.model.grid.shape[1], maxz)

        assert(len(nnz_sp_source_mask.dimensions) == (len(source_mask.dimensions)-1))

        # Note : sparse_source_id is not needed as long as sparse info is kept in mask
        # sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

        id_dim = Dimension(name='id_dim')
        b_dim = Dimension(name='b_dim')

        save_src_u = TimeFunction(name='save_src_u', shape=(src.shape[0],
                                  nzinds[1].shape[0]), dimensions=(src.dimensions[0],
                                  id_dim))
        save_src_v = TimeFunction(name='save_src_v', shape=(src.shape[0],
                                  nzinds[1].shape[0]), dimensions=(src.dimensions[0],
                                  id_dim))

        save_src_u_term = src.inject(field=save_src_u[src.dimensions[0], source_id],
                                     expr=src * self.model.grid.time_dim.spacing**2 / self.model.m)
        save_src_v_term = src.inject(field=save_src_v[src.dimensions[0], source_id],
                                     expr=src * self.model.grid.time_dim.spacing**2 / self.model.m)

        print("Injecting to empty grids")
        op1 = Operator([save_src_u_term, save_src_v_term])
        op1.apply(src=src, dt=kwargs.pop('dt', self.dt))
        print("Injecting to empty grids finished")
        sp_zi = Dimension(name='sp_zi')

        sp_source_mask = Function(name='sp_source_mask', shape=(list(sparse_shape)),
                                  dimensions=(x, y, sp_zi), space_order=0, dtype=np.int32)

        # Now holds IDs
        sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[-1][:len(np.unique(inds[-1]))])

        assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
        assert(len(sp_source_mask.dimensions) == 3)

        # import pdb; pdb.set_trace()         .

        zind = Scalar(name='zind', dtype=np.int32)
        xb_size = Scalar(name='xb_size', dtype=np.int32)
        yb_size = Scalar(name='yb_size', dtype=np.int32)
        x0_blk0_size = Scalar(name='x0_blk0_size', dtype=np.int32)
        y0_blk0_size = Scalar(name='y0_blk0_size', dtype=np.int32)

        block_sizes = Function(name='block_sizes', shape=(4, ), dimensions=(b_dim,),
                               space_order=0, dtype=np.int32)

        bsizes = (8, 8, 32, 32)
        block_sizes.data[:] = bsizes

        # eqxb = Eq(xb_size, block_sizes[0])
        # eqyb = Eq(yb_size, block_sizes[1])
        # eqxb2 = Eq(x0_blk0_size, block_sizes[2])
        # eqyb2 = Eq(y0_blk0_size, block_sizes[3])

        eq0 = Eq(sp_zi.symbolic_max, nnz_sp_source_mask[x, y] - 1,
                 implicit_dims=(time, x, y))
        # eq1 = Eq(zind, sp_source_mask[x, sp_zi], implicit_dims=(time, x, sp_zi))
        eq1 = Eq(zind, sp_source_mask[x, y, sp_zi], implicit_dims=(time, x, y, sp_zi))

        inj_u = source_mask[x, y, zind] * save_src_u[time, source_id[x, y, zind]]
        inj_v = source_mask[x, y, zind] * save_src_v[time, source_id[x, y, zind]]

        eq_u = Inc(u.forward[t+1, x, y, zind], inj_u, implicit_dims=(time, x, y, sp_zi))
        eq_v = Inc(v.forward[t+1, x, y, zind], inj_v, implicit_dims=(time, x, y, sp_zi))

        # The additional time-tiling equations
        # tteqs = (eqxb, eqyb, eqxb2, eqyb2, eq0, eq1, eq_u, eq_v)

        performance_map = np.array([[0, 0, 0, 0, 0]])

        bxstart = 4
        bxend = 17
        bystart = 4
        byend = 17
        bstep = 16

        txstart = 8
        txend = 9
        tystart = 8
        tyend = 9

        tstep = 16
        # Temporal autotuning
        for tx in range(txstart, txend, tstep):
            # import pdb; pdb.set_trace()
            for ty in range(tystart, tyend, tstep):
                for bx in range(bxstart, bxend, bstep):
                    for by in range(bystart, byend, bstep):

                        block_sizes.data[:] = [tx, ty, bx, by]

                        eqxb = Eq(xb_size, block_sizes[0])
                        eqyb = Eq(yb_size, block_sizes[1])
                        eqxb2 = Eq(x0_blk0_size, block_sizes[2])
                        eqyb2 = Eq(y0_blk0_size, block_sizes[3])

                        u.data[:] = 0
                        v.data[:] = 0
                        print("-----")
                        tteqs = (eqxb, eqyb, eqxb2, eqyb2, eq0, eq1, eq_u, eq_v)

                        op_tt = self.op_fwd(kernel, save, tteqs)
                        summary_tt = op_tt.apply(u=u, v=v,
                                                 dt=kwargs.pop('dt', self.dt), **kwargs)
                        norm_tt_u = norm(u)
                        norm_tt_v = norm(v)
                        print("Norm u:", regnormu)
                        print("Norm v:", regnormv)
                        print("Norm(tt_u):", norm_tt_u)
                        print("Norm(tt_v):", norm_tt_v)

                        print("===Temporal blocking======================================")

                        performance_map = np.append(performance_map, [[tx, ty, bx, by, summary_tt.globals['fdlike'].gflopss]], 0)


                print(performance_map)
                # tids = np.unique(performance_map[:, 0])

                #for tid in tids:
                bids = np.where((performance_map[:, 0] == tx) & (performance_map[:, 1] == ty))
                bx_data = np.unique(performance_map[bids, 2])
                by_data = np.unique(performance_map[bids, 3])
                gptss_data = performance_map[bids, 4]
                gptss_data = gptss_data.reshape(len(bx_data), len(by_data))

                fig, ax = plt.subplots()
                im = ax.imshow(gptss_data); pause(2)

                # We want to show all ticks...
                ax.set_xticks(np.arange(len(bx_data)))
                ax.set_yticks(np.arange(len(by_data)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(bx_data)
                ax.set_yticklabels(by_data)

                ax.set_title("Gpts/s for fixed tile size. (Sweeping block sizes)")
                fig.tight_layout()

                fig.colorbar(im, ax=ax)
                # ax = sns.heatmap(gptss_data, linewidth=0.5)
                plt.savefig(str(shape[0]) + str(np.int32(tx)) + str(np.int32(ty)) + ".pdf")


        if 0:
            cmap = plt.cm.get_cmap("viridis")
            values = u.data[0, :, :, :]
            vistagrid = pv.UniformGrid()
            vistagrid.dimensions = np.array(values.shape) + 1
            vistagrid.spacing = (1, 1, 1)
            vistagrid.origin = (0, 0, 0)  # The bottom left corner of the data set
            vistagrid.cell_arrays["values"] = values.flatten(order="F")
            vistaslices = vistagrid.slice_orthogonal()
            vistagrid.plot(show_edges=True)
            vistaslices.plot(cmap=cmap)

        return rec, u, v, summary

    def adjoint(self, rec, srca=None, p=None, r=None, vp=None,
                epsilon=None, delta=None, theta=None, phi=None,
                save=None, kernel='centered', **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        p : TimeFunction, optional
            The computed wavefield first component.
        r : TimeFunction, optional
            The computed wavefield second component.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        if kernel != 'centered':
            raise RuntimeError('Only centered kernel is supported for the adjoint')

        time_order = 2
        stagg_p = stagg_r = None
        # Source term is read-only, so re-use the default
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the wavefield if not provided
        if p is None:
            p = TimeFunction(name='p', grid=self.model.grid, staggered=stagg_p,
                             time_order=time_order,
                             space_order=self.space_order)
        # Create the wavefield if not provided
        if r is None:
            r = TimeFunction(name='r', grid=self.model.grid, staggered=stagg_r,
                             time_order=time_order,
                             space_order=self.space_order)

        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(self.model.physical_params(
            vp=vp, epsilon=epsilon, delta=delta, theta=theta, phi=phi)
        )
        if self.model.dim < 3:
            kwargs.pop('phi', None)
        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, p=p, r=r,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, p, r, summary
