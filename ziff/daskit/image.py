import dask
import dask.array as da

ZTF_IMAGE_SHAPE = (3080, 3072)

def get_ziff(file_, overwrite=False, waittime=None):
    """ """
    sciimg_mkimg = ziffit.get_file_delayed(file_, suffix=["sciimg.fits","mskimg.fits"],
                                               overwrite=overwrite, waittime=waittime,
                                               show_progress=False, maxnprocess=1)
    sciimg = sciimg_mkimg[0]
    mkimg  = sciimg_mkimg[1]
    #
    # - Build Ziff
    return base.ZIFF(sciimg, mkimg, fetch_psf=False)


def get_ziff_single_image(file_, which="data", overwrite=False, waittime=None, **kwargs):
    """ 
    Parameters
    ----------
    which: [string] -optional-
        which data do you want:

          // using self.get_data()
        - data: background subtracted images with NaN is masked pixels
        - raw: data as given (no background subtraction, no pixel masked)
        - rawmasked: data as given with NaN in masked pixel (= data+bkgd)

          // using self.get_background()
        - bkgd: background images

          // using self.get_mask()
        - mask: mask image (bool)
    """
    return get_ziff(file_, overwrite=overwrite, waittime=waittime).get_imagedata(which=which, **kwargs)

def get_ziff_single_background(file_, overwrite=False, waittime=None):
    """ """
    return get_ziff(file_, overwrite=overwrite, waittime=waittime).get_background()


def get_da_images(files, which="data", shape=ZTF_IMAGE_SHAPE, dtype="float32"):
    """ Get a dask.array stacked for each of the ziff image you want. 
    = Works only with single ziff = 
    """
    lazy_array = [dask.delayed(get_ziff_single_image)(f_, which=which) for f_ in files]
    lazy_arrays = [da.from_delayed(x_, shape=shape, dtype=dtype) for x_ in lazy_array]
    return da.stack(lazy_arrays)

def get_da_background(files, shape=ZTF_IMAGE_SHAPE, dtype="float32"):
    """ Get a dask.array stacked for each of the ziff image you want. 
    = Works only with single ziff = 
    """
    lazy_array = [dask.delayed(get_ziff_single_background)(f_)   for f_ in files]
    lazy_arrays = [da.from_delayed(x_, shape=shape, dtype=dtype) for x_ in lazy_array]
    return da.stack(lazy_arrays)

