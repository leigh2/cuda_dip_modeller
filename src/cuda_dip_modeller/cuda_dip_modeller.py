#!/usr/bin/env python3

import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
import os


# read the cuda source code
cu_src_file = os.path.join(os.path.dirname(__file__), "cuda_dip_modeller.cu")
with open(cu_src_file, "r") as f1:
    mod = SourceModule(f1.read())


# extract the kernels
_make_mugrid = mod.get_function("make_mugrid")
_populate_flux_points = mod.get_function("populate_flux_points")


def to_gpu(arr, dtype):
    """
    Convenience function for sending an array to the gpu with a specific data
    type.

    Parameters
    ----------
    arr : ndarray
        The array to send to the gpu
    dtype : dtype
        The numpy data type to use

    Returns
    -------
    A gpuarray
    """
    return gpuarray.to_gpu(arr.astype(dtype))


class dip_modeller(object):
    """

    """

    def __init__(self, ngrid, overgrid=10):
        """

        Parameters
        ----------
        ngrid
        overgrid
        """
        self.ngrid_full = np.int32(ngrid)
        self.overgrid = np.int32(overgrid)

        # initialise some arrays on the gpu
        self.xgrid_full = gpuarray.empty((ngrid, ngrid), dtype=np.float64)
        self.ygrid_full = gpuarray.empty((ngrid, ngrid), dtype=np.float64)
        self.mugrid_full = gpuarray.empty((ngrid, ngrid), dtype=np.float64)

        # block and grid sizes
        block_size = (32, 32, 1)
        grid_size = (
            int(np.ceil(self.ngrid_full / block_size[0])),
            int(np.ceil(self.ngrid_full / block_size[1]))
        )

        # populate the full grids
        # todo
        #  implement overbinning
        _make_mugrid(
            self.ngrid_full, self.overgrid,
            self.xgrid_full, self.ygrid_full, self.mugrid_full,
            block=block_size, grid=grid_size
        )

        # restrict to grid points containing stellar flux
        finite = np.isfinite(self.mugrid_full.get())
        self.ngrid = np.count_nonzero(finite)
        # filter and then send them back to the gpu
        self.xgrid = to_gpu(self.xgrid_full.get()[finite], np.float64)
        self.ygrid = to_gpu(self.ygrid_full.get()[finite], np.float64)
        self.mugrid = to_gpu(self.mugrid_full.get()[finite], np.float64)

    def __call__(self, xc, yc, transp, r0, rmaj, rmin, pa, alimb, blimb, xgrid, ygrid, mugrid):
        """

        Parameters
        ----------
        xc
        yc
        transp
        r0
        rmaj
        rmin
        pa
        alimb
        blimb
        xgrid
        ygrid
        mugrid

        Returns
        -------

        """
        self.n_lc_points = np.int32(xc.size)
        self.tmp_flux_pts = gpuarray.empty((self.n_lc_points, self.ngrid), dtype=np.float64)
