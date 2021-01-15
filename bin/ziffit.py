#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# ================ #
#                  #
#   MAIN           #
#                  #
# ================ #
if  __name__ == "__main__":

    import argparse
    from ztfquery import io
    from ziff import psffitter, __version__
    
    parser = argparse.ArgumentParser(
        description=""" Run the PSF extraction using the given files """,
        formatter_class=argparse.RawTextHelpFormatter)

    # On what file
    parser.add_argument('infile', type=str, default="None", nargs="*",
                        help='cube filepath')

    #
    # - PSF options
    #
    parser.add_argument('--interporder', type=int, default=3,
                        help='Order of the PSF spatial Interpolation.')

    parser.add_argument('--nstars', type=int, default=200,
                        help='Order of the PSF spatial Interpolation.')

    parser.add_argument('--boundpad', type=int, default=50,
                        help='Number of pixels around the CCD to be avoided.')

    #
    # - Options
    #
    parser.add_argument('--catalog', type=str, default="ps1cal",
                        help='Add the tag on output filename.')
    
    parser.add_argument('--addfilter', nargs=3, action='append',
                            help='Add a filter')

    #
    # - output
    #
    parser.add_argument('--noshapes', action="store_true", default=False,
                        help='No creation of the psf shape catalog file.')

    

    #
    # = ending
    args = parser.parse_args()

    #
    #
    #
    print("\nINFORMATION\n".center(80,'-'))
    print(f"* ziff version {__version__}")
    
    requested = args.infile
    print(f"* requested: {requested}")
    
    #
    science_file = [io.get_file(f_) for f_ in requested]
    print(f"     -> corresponding to {science_file}")

    # Catalog
    print(f"* Star catalog: {args.catalog}")

    #
    # - Filters
    addfilter = {f"{addfilter_[0]}_outrange":[addfilter_[0],
                                                      [float(addfilter_[1]),
                                                       float(addfilter_[2])]]
                    for addfilter_ in args.addfilter       
                }

    if len(addfilter)>0:
        print(f"* addition filter to apply on the catalog: {addfilter}")
    else:
        print("No additional catalog filtering")


    # ============== #
    #  SCRIPT        #
    # ============== #

    print("\nPSF FITTING\n".center(80,'-'))
    print(" == 1 == Loading ZIFFFitter")
    z = psffitter.ZIFFFitter(sciimg=requested, fetch_psf=False)

    print(f" == 2 == Fetching the *{args.catalog}* catalog")
    print("type(args.catalog): {type(args.catalog)}")
    if args.catalog in ["ps1cal"]:
        catalog = "ps1cal"
        z.fetch_ps1cal_catalog(name="ps1cal", bound_padding=args.boundpad)
    else:
        raise NotImplementedError("only ps1cal catalog has been implemented {args.catalog} given")

    print(f" == 3 == Setting the config options")
    z.set_nstars(args.nstars) # In general we only have ~200 calibrators / quadrant
    z.set_config_value('psf,interp,order', args.interporder)
    z.set_config_value('psf,outliers,max_remove',20)
    print(f" == 4 == Building the filtered catalog entiring PIFF ")
    cat = z.get_catalog(catalog, add_filter=addfilter, filtered=True)
    cat.change_name("ps1cal_tofit")
    print(f"         4.info: {len(cat.data)} stars in the catalog ; config limit: {args.nstars}")
    print(f" == 5 == Running piff ")
    z.run_piff(cat, on_filtered_cat=True)

    print("\nADDITIONAL OUTPUTS\n".center(80,'-'))
    if not args.noshapes:
        print(f" == 5.1 == Storing the PSF shapes.")
        z.store_psfshape(catalog, add_filter={"gmag_outmag":["gmag", [15,19]]})
        
    print("\nZIFFIT END\n".center(80,'-'))
