#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          run_ccd.py
# Description:       script description
# Author:            Romain Graziani <romain.graziani@clermont.in2p3.fr>
# Author:            $Author: rgraziani $
# Created on:        $Date: 2020/09/24 14:06:57 $
# Modified on:       2020/09/30 10:05:16
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
parser.add_argument("--interp_order",type=int,default=4)
parser.add_argument("--nstars",type=int,default=2000)



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
        z.set_config_value('i/o,nstars', 2000) 
        stars = z.make_stars('gaia_full',overwrite_cat=True, append_df_keys = ['RPmag','BPmag','colormag'])
        new_stars = z.reflux_stars(stars, fit_center = True, use_minuit = True)
        res = z.compute_residuals(new_stars)
        shapes = z.compute_shapes(new_stars,save=True)






# End of run_ccd.py ========================================================
