#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np

from ztfquery import io
from .. import base

import dask
#from .. import __version__


def limit_numpy(nthreads=4):
    """ """
    threads = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads

def _not_delayed_(func):
    return func

    
def ziffit_single(file_, use_dask=False, overwrite=False,
                      isolationlimit=10,
                      nstars=300, interporder=3, maxoutliers=None, 
                      fit_gmag=[15,16], shape_gmag=[15,19],
                      numpy_threads=None):
    """ """
    if numpy_threads is not None:
        limit_numpy(nthreads=numpy_threads)

    delayed = dask.delayed if use_dask else _not_delayed_

    #
    # - Get Files
    sciimg = delayed(io.get_file)(file_, suffix="sciimg.fits", overwrite=overwrite, 
                                      show_progress= not use_dask)
    mkimg  = delayed(io.get_file)(file_, suffix="mskimg.fits", overwrite=overwrite,
                                      show_progress= not use_dask)
    #
    # - Build Ziff    
    ziff   = delayed(base.ZIFF)(sciimg, mkimg, fetch_psf=False)
    #
    # - Get the catalog    
    cat   = delayed(base.get_gaia_catalog)(ziff, isolationlimit=isolationlimit,
                                                gmag_range=fit_gmag, shuffled=True)
    #
    # - Fit the PSF
    psf    = delayed(base.estimate_psf)(ziff, cat,
                                            interporder=interporder, nstars=nstars,
                                            maxoutliers=maxoutliers, verbose=False)
    #
    # - Compute shapes
    #catshp = delayed(base.get_gaia_catalog)(ziff, writeto="shape", gmag_range=shape_gmag,
    #                                            shuffled=True)
    # shapes
    shapes  = delayed(base.get_shapes)(ziff, psf, cat, store=True)
    
    return shapes[["sigma_model","sigma_data"]].median(axis=0).values
    
