

def limit_numpy(nthreads=4):
    """ """
    import os
    threads = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads

#limit_numpy(nthreads=4)



import piff
__version__="0.3.3"

from .models.pixelgridconvol import ConvolvedPixelGrid
piff.ConvolvedPixelGrid = ConvolvedPixelGrid



