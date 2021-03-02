#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
import time
import pandas
import numpy as np

from ztfquery import io
from .. import base

import dask
#from .. import __version__



def ziffit_single(file_, use_dask=False, overwrite=False,
                      isolationlimit=10, waittime=None,
                      nstars=300, interporder=3, maxoutliers=None,
                      stamp_size=15,
                      fit_gmag=[15,16], shape_gmag=[15,19],
                      numpy_threads=None):
    """ high level script function of ziff to 
    - find the isolated star from gaia 
    - fit the PSF using piff
    - compute and store the stars and psf-model shape parameters

    = Dask oriented =


    """
    if numpy_threads is not None:
        limit_numpy(nthreads=numpy_threads)

    delayed = dask.delayed if use_dask else _not_delayed_

    #
    # - Waiting time is any
    #
    # - Get Files

    # - This way, first sciimg, then mskimg, this enables not to overlead IRSA.
    sciimg_mkimg = delayed(get_file_delayed)(file_, waittime=waittime,
                                                 suffix=["sciimg.fits","mskimg.fits"],
                                                 overwrite=overwrite, 
                                                 show_progress= not use_dask, maxnprocess=1)
    sciimg = sciimg_mkimg[0]
    mkimg  = sciimg_mkimg[1]
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
    psf    = delayed(base.estimate_psf)(ziff, cat_toshape, stamp_size=stamp_size,
                                            interporder=interporder, nstars=nstars,
                                            maxoutliers=maxoutliers, verbose=False)
    # shapes
    shapes  = delayed(base.get_shapes)(ziff, psf, cat_tofit, stamp_size=stamp_size, store=True)
    
    return delayed(_get_ziffit_output_)(shapes)

def compute_shapes(file_, use_dask=False, numpy_threads=None, incl_residual=False, incl_stars=False):
    """ high level script function of ziff to 
    - compute and store the stars and psf-model shape parameters

    This re-perform the last steps of ziffit_single.

    = Dask oriented =

    """
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

    shapes       = delayed(base.get_shapes)(ziff, psf, cat_toshape, store=True,
                                                incl_residual=incl_residual, incl_stars=incl_stars)
    
    return shapes[["sigma_model","sigma_data"]].median(axis=0).values

    
def build_digitalized_shape(filenames, urange, vrange, chunks=50, nbins=200,
                            savefile=None, minimal=True, return_delayed=False, **kwargs):
    """ high level script function of ziff to 
    - read the computed shape parameters

    = Dask oriented =

    """
    filedf = get_filedataframe(filenames)
    grouped = filedf.groupby("filefracday")
    groupkeys = list( grouped.groups.keys() )
    
    bins_u = np.linspace(*urange, nbins)
    bins_v = np.linspace(*vrange, nbins)

    chunck_filenames = [np.concatenate([grouped["filename"].get_group(g_).values for g_ in chunk])
                            for chunk in np.array_split(groupkeys, chunks)]

    
    dfs = []
    for i, cfile in enumerate(chunck_filenames):
        dfs.append(dask.delayed(get_sigma_data)(cfile, bins_u, bins_v, minimal=minimal,
                                                savefile=None if savefile is None else savefile.replace(".parquet",f"_chunk{i}.parquet"),**kwargs
                                               )
                  )

    if return_delayed:
        return dfs
    
    dd = dask.compute(dfs)
    
    data = pandas.concat(dd[0])
    
    if savefile is not None:
        extension = savefile.split(".")
        if extension == "parquet":
            data.to_parquet(savefile)
        elif extension == "csv":
            data.to_csv(savefile)
        elif extension in ["hdf5", "hdf","h5"]:
            data.to_hdf(savefile)
        else:
            warnings.warn(f"Cannot store the file, unrecongized extension {extension}")
            
    return data

    
# ================ #
#    INTERNAL      #
# ================ #
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

def _get_ziffit_output_(shapes):
    """ """
    if shapes is not None:
        return shapes[["sigma_model","sigma_data"]].median(axis=0).values
    return [None,None]

def get_ziffit_gaia_catalog(ziff, isolationlimit=10,
                                fit_gmag=[15, 16], shape_gmag=[15,18],
                                shuffled=True, verbose=True):
    """ """
    if not ziff.has_images():
        warnings.warn("No image in the given ziff")
        return None,None

    try:
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
    except:
        warnings.warn("Failed grabing the gaia catalogs, Nones returned")
        return None,None


def get_sigma_data(files, bins_u, bins_v,
                    minimal=False,
                   quantity='sigma', normref="model", incl_residual=True,
                   basecolumns=['u', 'v', 'ccdid', 'qid', 'rcid', 'obsjd', 'fieldid','filterid', 'maglim'],
                   savefile=None,
                  ):
    if minimal:
        shape_columns = [f"{quantity}_data",  f"{quantity}_model"]
        incl_residual = False
    else:
        shape_columns = [f"sigma_data",f"sigma_model"] + \
                        [f"shapeg2_data",f"shapeg2_model"] + \
                        [f"shapeg1_data",f"shapeg1_model"] + \
                        [f"centerv_data",f"centerv_model"] + \
                        [f"centeru_data",f"centeru_model"]
    columns = basecolumns + shape_columns
    if incl_residual: 
        columns += ["residual"]

    filefracday = [f.split("/")[-1].split("_")[1] for f in files]
    df = pandas.concat([pandas.read_parquet(f, columns=columns) for f in files], keys=filefracday
                           ).reset_index().rename({"level_0":"filefracday"}, axis=1)
    
    norm = df.groupby(["obsjd"])[f"{quantity}_{normref}"].transform("median")
    
    df[f"{quantity}_data_n"] = df[f"{quantity}_data"]/norm
    df[f"{quantity}_model_n"] = df[f"{quantity}_model"]/norm
    df[f"{quantity}_residual"] = (df[f"{quantity}_data"]-df[f"{quantity}_model"])/df[f"{quantity}_model"]
    df["u_digit"] = np.digitize(df["u"],bins_u)
    df["v_digit"] = np.digitize(df["v"],bins_v)
    if savefile:
        df.to_parquet(savefile)
    return df


def get_filedataframe(files):
    from ztfquery import buildurl, fields
    
    def read_filename(file):
        _, filefracday,paddedfield, filtercode, ccdid_, _, qid_, suffix =  file.split("/")[-1].split("_")
        year,month, day, fracday = buildurl.filefrac_to_year_monthday_fracday(filefracday)
        ccdid = int(ccdid_[1:])
        qid = int(qid_[1:])        
        return {"obsdate": f"{year}-{month}-{day}",
                "fracday":fracday,
                "filefracday":filefracday,
                "fieldid":int(paddedfield),
                "filtername":filtercode,
                "ccdid":ccdid,
                "qid":qid,
                "rcid":fields.ccdid_qid_to_rcid(ccdid,qid),
                "suffix":suffix,
                "filename":file
               }

    return pandas.DataFrame([read_filename(f) for f in files])


def get_file_delayed(file_, waittime=None,
                         suffix=["sciimg.fits","mskimg.fits"], overwrite=False, 
                         show_progress=True, maxnprocess=1, **kwargs):
    """ """
    if waittime is not None:
        time.sleep(waittime)
        
    return io.get_file(file_, waittime=waittime,
                        suffix=suffix, overwrite=overwrite, 
                        show_progress=show_progress, maxnprocess=maxnprocess)
    
# ================ #
#    INTERNAL      #
# ================ #
