""" Tools to build and PSF Cluster based """


from ztfquery import io
from .. import base

def get_ziff_psf_cat(file_, whichpsf="psf_PixelGrid_BasisPolynomial5.piff"):
    """ """
    files_needed = io.get_file(file_, suffix=[whichpsf,"sciimg.fits", "mskimg.fits",
                                              "shapecat_gaia.fits"], check_suffix=False)
    # Dask
    psffile, sciimg, mkimg, catfile = files_needed[0],files_needed[1],files_needed[2],files_needed[3]

    ziff         = base.ZIFF(sciimg, mkimg, fetch_psf=False)
    cat_toshape  = base.catlib.Catalog.load(catfile, wcs=ziff.wcs)
    psf          = base.piff.PSF.read(file_name=psffile, logger=None)
    
    return ziff, psf, cat_toshape



class ClusterZiffit( object ):
    """ """
    
