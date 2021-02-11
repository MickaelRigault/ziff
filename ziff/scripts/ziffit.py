#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
import time
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


def get_ziffit_gaia_catalog(ziff, isolationlimit=10,
                                fit_gmag=[15, 16], shape_gmag=[15,18],
                                shuffled=True, verbose=True):
    """ """
    if "gaia" not in ziff.catalog:
        if verbose:
            print("loading gaia")
        ziff.fetch_gaia_catalog(isolationlimit=isolationlimit)
        
    cat_to_fit   = ziff.get_catalog("gaia", filtered=True, shuffled=shuffled, 
                              writeto="default",
                              add_filter={'gmag_outrange':['gmag', fit_gmag]},
                              xyformat="fortran")
    
    cat_to_shape = ziff.get_catalog("gaia", filtered=True, shuffled=shuffled, 
                              writeto="shape",
                              add_filter={'gmag_outrange':['gmag', shape_gmag]},
                              xyformat="fortran")
    
    return cat_to_fit,cat_to_shape


def get_file_wait(file_, waittime=None,
                    suffix=None, overwrite=False, show_progress=True, **kwargs):
    """ """
    if waittime is None:
        time.sleep(waittime)
        
    return io.get_file(file_, suffix="sciimg.fits", overwrite=overwrite, 
                           show_progress=show_progress, **kwargs)

def ziffit_single(file_, use_dask=False, overwrite=False,
                      isolationlimit=10, waittime=None,
                      nstars=300, interporder=3, maxoutliers=None, 
                      fit_gmag=[15,16], shape_gmag=[15,19],
                      numpy_threads=None):
    """ """
    if numpy_threads is not None:
        limit_numpy(nthreads=numpy_threads)

    delayed = dask.delayed if use_dask else _not_delayed_

    #
    # - Get Files
    sciimg = delayed(get_file_wait)(file_, waittime=waittime,
                                        suffix="sciimg.fits", overwrite=overwrite, 
                                        show_progress= not use_dask)
    mkimg  = delayed(get_file_wait)(file_, waittime=waittime,
                                        suffix="mskimg.fits", overwrite=overwrite,
                                        show_progress= not use_dask)
    #
    # - Build Ziff    
    ziff   = delayed(base.ZIFF)(sciimg, mkimg, fetch_psf=False)
    #
    # - Get the catalog    
    cats  = delayed(get_ziffit_gaia_catalog)(ziff, fit_gmag=fit_gmag, shape_gmag=shape_gmag,
                                                 isolationlimit=isolationlimit,shuffled=True)
    cat_tofit  = cats[0]
    cat_toshape= cats[1]
    #
    # - Fit the PSF
    psf    = delayed(base.estimate_psf)(ziff, cat_toshape,
                                            interporder=interporder, nstars=nstars,
                                            maxoutliers=maxoutliers, verbose=False)
    # shapes
    shapes  = delayed(base.get_shapes)(ziff, psf, cat_tofit, store=True)
    
    return shapes[["sigma_model","sigma_data"]].median(axis=0).values
    


def compute_shapes(file_, use_dask=False, numpy_threads=None):
    """ """
    if numpy_threads is not None:
        limit_numpy(nthreads=numpy_threads)

    delayed = dask.delayed if use_dask else _not_delayed_


    files_needed = delayed(io.get_file)(file_, suffix=["psf_PixelGrid_BasisPolynomial3.piff", 
                                                       "sciimg.fits", "mskimg.fits",
                                                       "shapecat_gaia.fits"], check_suffix=False)
    # Dask
    psffile, sciimg, mkimg, catfile = files_needed[0],files_needed[1],files_needed[2],files_needed[3]

    ziff         = delayed(base.ZIFF)(sciimg, mkimg, fetch_psf=False)
    cat_toshape  = delayed(base.catlib.Catalog.load)(catfile, wcs=ziff.wcs)
    psf          = delayed(base.piff.PSF.read)(file_name=psffile, logger=None)

    shapes       = delayed(base.get_shapes)(ziff, psf, cat_toshape, store=True)
    
    return shapes[["sigma_model","sigma_data"]].median(axis=0).values

    
