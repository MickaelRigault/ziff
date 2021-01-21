#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .. import __version__

from dask import delayed


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

# ================= #
#  SCRIPT STEPS     #
# ================= #

def get_file(filename, dlfrom="irsa", allowdl=True, verbose=True, **kwargs):
    """ """
    #
    if verbose:
        print(" == 1 == Grabbing Files")
        print(f"* requested: {filename}")

    from ztfquery import io
    prop = {**dict(dlfrom=dlfrom, downloadit=allowdl), **kwargs}
    sciimg_files = [io.get_file(f_, suffix="sciimg.fits",**prop) for f_ in np.atleast_1d(filename)]
    mskimg_files = [io.get_file(f_, suffix="mskimg.fits", **prop) for f_ in np.atleast_1d(filename)]

    if verbose:
        print(f"\t-> corresponding to \n\t{sciimg_files} \n\tand\n\t{mskimg_files}")
        
    return sciimg_files, mskimg_files

def get_ziff(sciimg_files, mskimg_files=None, logger=None, fetch_psf=False, config="default",
                 verbose=True):
    """ """
    if verbose:
        print(" == 2 == Loading ZIFF")
        
    from ziff.psffitter import ZIFFFitter
    return ZIFFFitter(sciimg=sciimg_files,mskimg=mskimg_files,
                        logger=logger, fetch_psf=fetch_psf, config=config)

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

        
    if catalog in ["ps1cal"]:
        catalog = "ps1cal"
        ziff.fetch_ps1cal_catalog(name="ps1cal", bound_padding=boundpad, isolationlimit=isolationlimit)
    else:
        raise NotImplementedError("only ps1cal catalog has been implemented {args.catalog} given")

    cat_to_fit = ziff.get_catalog(catalog, add_filter=add_filter, xyformat=xyformat, filtered=filtered)
    cat_to_fit.change_name(f"{catalog}_tofit")
    return cat_to_fit

def run_piff(ziff, catalog, nstars=300, interporder=3, maxoutliers=30,
                 verbose=True):
    """ """
    if verbose:
        print(" == 4 == Running PIFF")
    ziff.set_nstars(nstars) # In general we only have ~200 calibrators / quadrant
    ziff.set_config_value('psf,interp,order', interporder)
    ziff.set_config_value('psf,outliers,max_remove',maxoutliers)
    
    return ziff.run_piff(catalog, on_filtered_cat=True, verbose=verbose)


def store_psfshape(ziff, catalog, psf=None, addfilter=None,
                      verbose=True, getshape=True):
    """ """
    if verbose: print(f" Storing the PSF shapes.")
    add_filter = _parse_addfilters_(addfilter)
    return ziff.store_psfshape(catalog, psf=psf, add_filter=add_filter, getshape=getshape)
    

def checkout_ziffit(psf, shapes):
    """ """
    # Add test here
    test = "ok"
    return test == "ok"
    
# ================= #
#    MAIN           #
# ================= #
def dask_ziffit(files, catalog="ps1cal", 
                fit_filter=[["gmag",14,16]],
                shape_catfilter=[["gmag",14,19]],
                **kwargs):
    """ """
    works = []
    for i,file_ in enumerate(files):
        worked_psf_shapes = dask_single(file_, catalog=catalog,
                         fit_filter=fit_filter,
                         shape_catfilter=shape_catfilter,
                         **kwargs)
        works.append(worked[0])
        
    return delayed(np.asarray)(works) # This is useless but final point


def dask_single(file_, catalog="ps1cal", verbose=False,
                dlfrom="irsa", allowdl=True,
                logger=None, fetch_psf=False, config="default",
                boundpad=50,
                fit_filter=[["gmag",14,16]],
                shape_catfilter=[["gmag",14,19]],
                fit_isolationlimit=8, shape_isolationlimit=8,
                nstars=300, interporder=3, maxoutliers=30):
    """ """
    fitimages  = delayed(get_file)(file_, dlfrom=dlfrom, 
                                       allowdl=allowdl, verbose=verbose, show_progress=False)

    ziff       = delayed(get_ziff)(fitimages[0], fitimages[1],
                                 logger=logger, fetch_psf=fetch_psf, 
                                config=config, verbose=verbose)

    cat_to_fit = delayed(get_catalog)(ziff, catalog,
                                          boundpad=boundpad, addfilter=fit_filter,
                                          isolationlimit=fit_isolationlimit,
                                          verbose=verbose)

    psf       = delayed(run_piff)(ziff, cat_to_fit, 
                                      nstars=nstars, interporder=interporder, 
                                      maxoutliers=maxoutliers,
                                      verbose=verbose)
    
    cat_shape = delayed(get_catalog)(ziff, catalog,
                                         boundpad=boundpad, 
                                         addfilter=shape_catfilter,
                                        isolationlimit=shape_isolationlimit,
                                         verbose=verbose)

    shapes   = delayed(store_psfshape)(ziff, cat_shape, psf=psf,  getshape=True)
    
    worked   = delayed(checkout_ziffit)(psf, shapes)
    
    return psf


def single(file_, catalog="ps1cal", verbose=False,
                dlfrom="irsa", allowdl=True,
                logger=None, fetch_psf=False, config="default",
                boundpad=50,
                fit_filter=[["gmag",14,16]],
                shape_catfilter=[["gmag",14,19]],
                fit_isolationlimit=8, shape_isolationlimit=8,
                nstars=300, interporder=3, maxoutliers=30):
    """ """
    fitimages  = get_file(file_, dlfrom=dlfrom, 
                                       allowdl=allowdl, verbose=verbose, show_progress=False)

    ziff       = get_ziff(fitimages[0], fitimages[1],
                                 logger=logger, fetch_psf=fetch_psf, 
                                config=config, verbose=verbose)

    cat_to_fit = get_catalog(ziff, catalog,
                                          boundpad=boundpad, addfilter=fit_filter,
                                          isolationlimit=fit_isolationlimit,
                                          verbose=verbose)

    psf       = run_piff(ziff, cat_to_fit, 
                                      nstars=nstars, interporder=interporder, 
                                      maxoutliers=maxoutliers,
                                      verbose=verbose)
    
    cat_shape = get_catalog(ziff, catalog,
                                         boundpad=boundpad, 
                                         addfilter=shape_catfilter,
                                        isolationlimit=shape_isolationlimit,
                                         verbose=verbose)

    shapes   = store_psfshape(ziff, cat_shape, psf=psf,  getshape=True)
    
    worked   = checkout_ziffit(psf, shapes)
    
    return psf


    

def ziffit(filenames, catalog="ps1cal", verbose=True,
               dlfrom="irsa", allowdl=True,
               logger=None, fetch_psf=False, config="default",
               boundpad=50, addfilter=None, fit_isolationlimit=None,
               nstars=300, interporder=3, maxoutliers=30,
               store_shapes=True):
    """ """
    if verbose:
        print("\ZIFFIT PSF FITTING STARTS \n".center(80,'-'))
        print(f"* ziff version {__version__}")
    
    # - Step 1
    sciimg_files, mskimg_files = get_file(filenames,
                                            dlfrom=dlfrom, allowdl=allowdl,
                                              verbose=verbose)

    # - Step 2
    ziff  = get_ziff(sciimg_files, mskimg_files,
                         logger=logger, fetch_psf=fetch_psf, config=config,
                         verbose=verbose)

    # - Step 3
    cat_to_fit = get_catalog(ziff, catalog,
                                 boundpad=boundpad, addfilter=addfilter,
                                 isolationlimit=fit_isolationlimit,
                                 verbose=verbose)

    # - Step 4
    run_piff(ziff, cat_to_fit, nstars=nstars, interporder=interporder, maxoutliers=maxoutliers,
                 verbose=verbose)
    

    # - Step 5, additional outputs:
    print("\nADDITIONAL OUTPUTS\n".center(80,'-'))
    if store_shapes:
        store_psfshape(ziff, catalog, addfilter=[["gmag", 15,19]])

    if verbose:
        print("\nZIFFIT END\n".center(80,'-'))
