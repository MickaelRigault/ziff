#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          download_query.py
# Description:       script description
# Author:            Romain Graziani <romain.graziani@clermont.in2p3.fr>
# Author:            $Author: rgraziani $
# Created on:        $Date: 2020/09/25 16:23:01 $
# Modified on:       2020/09/28 10:17:09
# Copyright:         2019, Romain Graziani
# $Id: download_query.py, 2020/09/25 16:23:01  RG $
################################################################################

"""
.. _download_query.py:

download_query.py
==============


"""
__license__ = "2019, Romain Graziani"
__docformat__ = 'reStructuredText'
__author__ = 'Romain Graziani <romain.graziani@clermont.in2p3.fr>'
__date__ = '2020/09/25 16:23:01'
__adv__ = 'download_query.py'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--query",type=str,default = "obsjd BETWEEN 2458554.5 AND 2458564")
parser.add_argument("--overwrite",type=int,default = 1)
parser.add_argument("--nprocess",type=int,default = 1)


args = parser.parse_args()

in_query = args.query
from ztfquery import query
print(in_query)
zquery = query.ZTFQuery() 
zquery.load_metadata(sql_query = in_query)
print(zquery.metatable)
keys = ['sciimg.fits', 'mskimg.fits', 'psfcat.fits']
for _key in keys:
    zquery.download_data(_key,show_progress=True, notebook=False, nprocess=args.nprocess, overwrite = bool(args.overwrite))
        

# End of download_query.py ========================================================
