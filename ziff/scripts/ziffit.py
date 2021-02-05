#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .. import __version__
import warnings

from dask import delayed

from ztfquery import io
from ziff.base import ZIFF


def _parse_addfilters_(addfilter):
    """ """
    if addfilter is None:
        return None
    
    addfilter = np.atleast_1d(addfilter)
    return {f"{addfilter_[0]}_outrange":[addfilter_[0],
                                             [float(addfilter_[1]),
                                              float(addfilter_[2])]]
                    for addfilter_ in addfilter       
            }

def _not_delayed_(func):
    return func
# ================= #
#  SCRIPT STEPS     #
# ================= #
def get_ziff(sciimg_files, mskimg_files=None, logger=None, fetch_psf=False, config="default",
                 verbose=True, load_background=True):
    """ """
    if verbose:
        print(" == 2 == Loading ZIFF")
        
    ziff_ = ZIFF(sciimg=sciimg_files,mskimg=mskimg_files,
                        logger=logger, fetch_psf=fetch_psf, config=config)
    if load_background:
        ziff_.load_image_sourcebackground()
        
    return ziff_
    
def get_catalog(ziff, catalog, boundpad=50, addfilter=None, filtered=True, xyformat=None,
                    isolationlimit=None,
                    verbose=True):
    """ """
    if verbose:
        print(" == 3 == Loading the fit Catalog")
        print(f"* requesting {catalog} catalog")
        
    add_filter = _parse_addfilters_(addfilter)

    if len(add_filter)>0:
        print(f"* addition filter to apply on the catalog: {add_filter}")
    else:
        print("No additional catalog filtering")
        
    if catalog not in ["ps1cal", "gaia"]:
        raise NotImplementedError(f"only ps1cal and gaia catalog have been implemented {catalog} given")
    # Fetch only if necessary
    elif catalog in ["ps1cal"] and "ps1cal" not in ziff.catalog:
        ziff.fetch_ps1cal_catalog(name="ps1cal", bound_padding=boundpad,
                                      isolationlimit=isolationlimit)
    elif catalog in ["gaia"] and "gaia" not in ziff.catalog:
        ziff.fetch_gaia_catalog(name="gaia", bound_padding=boundpad,
                                    isolationlimit=isolationlimit)

    cat_to_fit = ziff.get_catalog(catalog, add_filter=add_filter, xyformat=xyformat,
                                      filtered=filtered)
    cat_to_fit.change_name(f"{catalog}_tofit")
    return cat_to_fit

def run_piff(ziff, catalog, minstars=30, nstars=300, interporder=3, maxoutliers=30,
                 verbose=True):
    """ """
    if verbose:
        print(" == 4 == Running PIFF")
    ziff.set_nstars(nstars) # In general we only have ~200 calibrators / quadrant
    ziff.set_config_value('psf,interp,order', interporder)
    ziff.set_config_value('psf,outliers,max_remove',maxoutliers)
    
    return ziff.run_piff(catalog, minstars=minstars,
                             on_filtered_cat=True, verbose=verbose)


def store_psfshape(ziff, catalog, psf, addfilter=None,
                      verbose=True, getshape=True):
    """ 
    Parameters
    ----------
    psf: [piff.PSF or None]
        if psf is None this returns None (Dask safe)

    """
    if verbose: print(f" Storing the PSF shapes.")
    add_filter = _parse_addfilters_(addfilter)
    nopsf = psf is None
    
    return ziff.store_psfshape(catalog, psf=psf, add_filter=add_filter,
                                   getshape=getshape, nopsf=nopsf)
    

def checkout_ziffit(psf, shapes):
    """ """
    # Add test here
    if psf is None or shapes is None:
        worked = False
    else:
        worked = True
        
    return worked
    
# ================= #
#    MAIN           #
# ================= #
def ziffit(files, catalog="gaia", use_dask=True,
            fit_filter=[["Gmag",14,16]],
            shape_catfilter=[["Gmag",14,19]],
            **kwargs):
    """ """
    psfs = []
    for i,file_ in enumerate(files):
        psf_ = ziffit_single(file_, catalog=catalog, use_dask=use_dask,
                                 fit_filter=fit_filter,
                                 shape_catfilter=shape_catfilter,
                                 **kwargs)
        psfs.append(psf_)
        
    return psfs # This is useless but final point


def ziffit_single(file_, catalog="gaia", verbose=False,
                      use_dask=True,
                dlfrom="irsa", allowdl=True, overwrite=True,
                logger=None, fetch_psf=False, config="default",
                boundpad=50, 
                fit_filter=[["Gmag",14,16]],
                shape_catfilter=[["Gmag",14,19]],
                fit_isolationlimit=8, shape_isolationlimit=8,
                piffit=True, getshape=True,
                minstars=30, nstars=300, interporder=3, maxoutliers=30):
    """ """
    delayedfunc = delayed if use_dask else _not_delayed_

    
    fitsciimg  = delayedfunc(io.get_file)(file_, suffix="sciimg.fits", maxnprocess=1, dlfrom=dlfrom, 
                                        overwrite=overwrite, downloadit=allowdl,
                                        verbose=verbose, show_progress=False, squeez=True)
    
    fitmskimg  = delayedfunc(io.get_file)(file_, suffix="mskimg.fits", maxnprocess=1, dlfrom=dlfrom, 
                                        overwrite=overwrite, downloadit=allowdl,
                                        verbose=verbose, show_progress=False, squeez=True)
    
    ziff       = delayedfunc(get_ziff)(fitsciimg, fitmskimg,
                                    logger=logger, fetch_psf=fetch_psf, 
                                    config=config, verbose=verbose)

    cat_to_fit = delayedfunc(get_catalog)(ziff, catalog, 
                                        boundpad=boundpad, addfilter=fit_filter,
                                        isolationlimit=fit_isolationlimit,
                                        verbose=verbose)
    
    if not piffit:
        return cat_to_fit.npoints
    
    psf       = delayedfunc(run_piff)(ziff, cat_to_fit, 
                                      minstars=minstars, nstars=nstars, interporder=interporder, 
                                      maxoutliers=maxoutliers,
                                     verbose=verbose)
    if not getshape:
        return psf

    cat_shape = delayedfunc(get_catalog)(ziff, catalog, 
                                         boundpad=boundpad, 
                                         addfilter=shape_catfilter,
                                        isolationlimit=shape_isolationlimit,
                                         verbose=verbose)

    shapes   = delayedfunc(store_psfshape)(ziff, cat_shape, psf=psf,  getshape=True)
    
    # worked   = delayed(checkout_ziffit)(psf, shapes)
        
    # - output
    return shapes["sigma_stars"].mean()
