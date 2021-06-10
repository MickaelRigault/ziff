

from ztfquery import io
from . import base
import dask

DEFAULT_FIT_GMAG   = [13, 18]
DEFAULT_SHAPE_GMAG = [13, 18]
DEFAULT_ISOLATION  = 20


class ZiffDask( object ):
    """ """

    @staticmethod
    def get_file(filename, waittime=None, **kwargs):
        """ """
        prop = {**dict(overwrite=False, show_progress=False, maxnprocess=1),
                **kwargs}
            
        return dask.delayed(io.get_file)(filename,
                                         waittime=waittime,
                                         suffix=["sciimg.fits","mskimg.fits"],
                                         **prop)
    
    @classmethod
    def compute_single(cls, filename, catisolation=DEFAULT_ISOLATION,
                           fit_gmag=DEFAULT_FIT_GMAG,
                           shape_gmag=DEFAULT_SHAPE_GMAG, shufflecat=True
                           stamp_size=17, interporder=5, nstars=300,
                           maxoutliers=None, **kwargs):
        """ """
        # Get the target
        sciimg_mkimg = cls.get_file(filename, **kwargs)
        sciimg = sciimg_mkimg[0]
        mkimg  = sciimg_mkimg[1]

        #
        # Building ZIFF
        ziff   = dask.delayed(base.ZIFF)(sciimg, mkimg, fetch_psf=False)
        
        #
        # Building the catalogs
        cat_tofit = ziff.get_gaia_catalog( isolation=catisolation,
                                            gmag_range=fit_gmag,
                                            shuffled=shufflecat, writeto="psf", 
                                            xyformat="fortran")

        cat_toshape = ziff.get_gaia_catalog( isolation=catisolation,
                                            gmag_range=shape_gmag,
                                            shuffled=shufflecat, writeto="shape", 
                                            xyformat="fortran")
        #
        # Fitting the PSF model
        psf    = delayed(base.estimate_psf)(ziff, cat_tofit, stamp_size=stamp_size,
                                            interporder=interporder, nstars=nstars,
                                            maxoutliers=maxoutliers, verbose=False)
        # Shapes
        shapes  = delayed(base.get_shapes)(ziff, psf, cat_toshape, store=True,
                                            stamp_size=stamp_size,
                                            incl_residual=True, incl_stars=True)
        return psf, shapes
