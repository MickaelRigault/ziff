#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          run_ccd.py
# Description:       script description
# Author:            Romain Graziani <romain.graziani@clermont.in2p3.fr>
# Author:            $Author: rgraziani $
# Created on:        $Date: 2020/09/24 14:06:57 $
# Modified on:       2020/10/01 10:12:17
# Copyright:         2019, Romain Graziani
# $Id: run_ccd.py, 2020/09/24 14:06:57  RG $
################################################################################

"""
.. _run_ccd.py:

run_ccd.py
==============


"""
__license__ = "2019, Romain Graziani"
__docformat__ = 'reStructuredText'
__author__ = 'Romain Graziani <romain.graziani@clermont.in2p3.fr>'
__date__ = '2020/09/24 14:06:57'
__adv__ = 'run_ccd.py'

import ziff.ziff
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

import argparse
import pkg_resources

parser = argparse.ArgumentParser()
parser.add_argument("--filename",type=str)
parser.add_argument("--rows",type=int, nargs='+')
parser.add_argument("--make_cats",type=int,default=1)
parser.add_argument("--run",type=int,default=1)
parser.add_argument("--shapes",type=int,default=1)
parser.add_argument("--plot",type=int,default=1)
parser.add_argument("--interp_order",type=int,default=4)
parser.add_argument("--nstars",type=int,default=2000)
parser.add_argument("--shape_nstars",type=int,default=2000)




args = parser.parse_args()
rows = np.atleast_1d(args.rows)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for row in rows:
    if args.make_cats:
        z = ziff.ziff.Ziff.from_file(args.filename, row = row, build_default_cat = True, load_default_cat = False, save_cat = True)
    else:
        z = ziff.ziff.Ziff.from_file(args.filename, row = row, build_default_cat = False, load_default_cat = True, save_cat = False)
    z.set_config_value('psf,interp,order', args.interp_order)
    z.set_config_value('i/o,nstars', args.nstars) 
    z.set_config_value('psf,outliers,max_remove',20)
    #map_file = pkg_resources.resource_filename('ziff', 'data/interpolator.pkl')
    #z.set_config_value('psf,interp,interpolation_map_file',map_file)
    #z.set_config_value('psf,interp,type','BasisPolynomialPlusMap')
    if args.run:
        z.run_piff('gaia_calibration',overwrite_cat=True)
    if args.shapes:
        z.set_config_value('i/o,nstars', args.shape_nstars) 
        stars = z.make_stars('gaia_full',overwrite_cat=False, append_df_keys = ['RPmag','BPmag','colormag'])
        new_stars = z.reflux_stars(stars, fit_center = True, use_minuit = True)
        res = z.compute_residuals(new_stars)
        shapes = z.compute_shapes(new_stars,save=True)

        if args.plot:
            import matplotlib.pyplot as P
            im_kwargs  = {'origin':'lower', 'vmin' : -0.03, 'vmax': 0.03}
            fig, axes = P.subplots(1,3,figsize=(6,2))
            axes[0].imshow(res[0].T, **im_kwargs)
            axes[1].imshow(np.mean(res,axis=0).T, **im_kwargs)
            axes[2].imshow(np.median(res,axis=0).T, **im_kwargs)
            for p in z.prefix:
                fig.savefig(p + 'psf_piff_res.pdf')
            fig, axes = P.subplots(1,3,figsize=(12,3))
            scat_kwargs = {'cmap':'RdBu_r', 's':50}
            s = axes[0].scatter(shapes['u'],shapes['v'],c=np.asarray(shapes['T_data_normalized']),vmin=0.9,vmax=1.1,**scat_kwargs)
            fig.colorbar(s,ax=axes[0])
            s = axes[1].scatter(shapes['u'],shapes['v'],c=np.asarray(shapes['T_model_normalized']),vmin=0.9,vmax=1.1,**scat_kwargs)
            fig.colorbar(s,ax=axes[1])
            s = axes[2].scatter(shapes['u'],shapes['v'],c=np.asarray(shapes['T_data'])-np.asarray(shapes['T_model']),vmin=-0.05,vmax=0.05,**scat_kwargs)
            fig.colorbar(s,ax=axes[2])
            for p in z.prefix:
                fig.savefig(p + 'psf_piff_shape.pdf')

        





# End of run_ccd.py ========================================================
