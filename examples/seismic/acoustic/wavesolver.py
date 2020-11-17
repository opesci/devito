from devito import Function, TimeFunction, norm, Operator, Dimension, Scalar, Eq, Inc, configuration
from devito.tools import memoized_meth
from examples.seismic.acoustic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from devito.types.basic import Scalar
from matplotlib.pyplot import pause # noqa
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

class AcousticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    kernel : str, optional
        Type of discretization, centered or shifted.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, kernel='OT2', space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == 'OT4':
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None, tteqs=(), **kwargs):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               tteqs=tteqs, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                **self._kwargs)

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, geometry=self.geometry,
                            kernel=self.kernel, space_order=self.space_order,
                            **self._kwargs)

    def forward(self, src=None, rec=None, u=None, vp=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            Stores the computed wavefield.
        vp : Function or float, optional
            The time-constant velocity.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        # summary = self.op_fwd(save).apply(src=src, rec=rec, u=u, vp=vp,
        summary = self.op_fwd(save).apply(src=src, u=u, vp=vp,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)

        regnormu = norm(u)
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

        print("Norm u:", regnormu)

        s_u = TimeFunction(name='s_u', grid=self.model.grid, space_order=self.space_order, time_order=1)
        src_u = src.inject(field=s_u.forward, expr=src * self.model.grid.time_dim.spacing**2 / self.model.m)


        op_f = Operator([src_u])
        op_f.apply(src=src, dt=kwargs.pop('dt', self.dt))

        print("Norm s_u", norm(s_u))

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

        print("Number of unique affected points is:", len(nzinds[0]))

        # Assert that first and last index are as expected
        assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
        assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
        assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1], nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

        assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
        assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(s_u.data[0])))

        print("-At this point source_mask and source_id have been populated correctly-")

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

        # import pdb;pdb.set_trace()

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

        eq_u = Inc(u.forward[t+1, x, y, zind], inj_u, implicit_dims=(time, x, y, sp_zi))

        # The additional time-tiling equations
        # tteqs = (eqxb, eqyb, eqxb2, eqyb2, eq0, eq1, eq_u, eq_v)

        performance_map = np.array([[0, 0, 0, 0, 0]])

        bxstart = 8
        bxend = 17
        bystart = 8
        byend = 17
        bstep = 16

        txstart = 32
        txend = 33
        tystart = 32
        tyend = 33

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
                        # v.data[:] = 0
                        print("-----")
                        tteqs = (eqxb, eqyb, eqxb2, eqyb2, eq0, eq1, eq_u)

                        # import pdb; pdb.set_trace()

                        # Execute operator and return wavefield and receiver data
                        summary_tt = self.op_fwd(save, tteqs).apply(u=u, vp=vp,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)

                        # op_tt = self.op_fwd(save, tteqs)

                        # Execute operator and return wavefield and receiver data
                        #summary_tt = self.op_fwd(save).apply(src=src, rec=rec, u=u, vp=vp,
                        #                                     dt=kwargs.pop('dt', self.dt), **kwargs)

                        # op_tt = self.op_fwd(kernel, save, tteqs)
                        # summary_tt = op_tt.apply(u=u, dt=kwargs.pop('dt', self.dt), **kwargs)
                        configuration['jit-backdoor'] = False
                        norm_tt_u = norm(u)
                        print("Norm u:", regnormu)
                        print("Norm(tt_u):", norm_tt_u)
                        configuration['jit-backdoor'] = True

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


        return rec, u, summary

    def adjoint(self, rec, srca=None, v=None, vp=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, v=v, vp=vp,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    def jacobian_adjoint(self, rec, u, v=None, grad=None, vp=None,
                         checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.geometry.src,
                                         u=u, vp=vp, dt=dt)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          vp=vp, rec=rec, dt=dt, grad=grad)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u, vp=vp,
                                           dt=dt, **kwargs)
        return grad, summary

    def jacobian(self, dmin, src=None, rec=None, u=None, U=None, vp=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        U : TimeFunction, optional
            The linearized wavefield.
        vp : Function or float, optional
            The time-constant velocity.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                       vp=vp, dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary

    # Backward compatibility
    born = jacobian
    gradient = jacobian_adjoint
