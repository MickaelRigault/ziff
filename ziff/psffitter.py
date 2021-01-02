#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Applying the PIFF PSF software (Jarvis et al.) on ZTF images """

import os
import numpy as np
import pandas as pd
import json

from astropy.io import fits
from astropy.wcs import WCS

# - PIFF
import piff
from piff.star import Star, StarData, StarFit

# - zfquery / ztfimg
from ztfimg import image

# - LOCAL
from . import catalog

def get_stars_cat_kwargs(stars):
    """ """
    out = {}
    keys = stars[0]._cat_kwargs.keys()
    for k in keys:
        out[k] = []
    for s in stars:
        for k in keys:
            out[k].append(s._cat_kwargs[k])
    return out

######################
#                    #
#  Ziff Class        #
#                    #
######################
from .base import ZIFF

class ZIFFFitter( ZIFF ):
    # Wrapper of piff for ztf
    

    # ================ #
    #   Methods        #
    # ================ #
    # --------- #
    #  PIFF     #
    # --------- #
    def run_piff(self, catalog="default", catfileout=None, overwrite_cat=True,
                     fitcatprop={}, 
                     on_filtered_cat=True,
                     save_suffix='output.piff'):
        """ run the piff PSF algorithm on the given images using 
        the given reference catalog (star location) 
        
        Parameters
        ----------
        catalog
        """
        # 0.
        # - parse the input catalog
        if type(catalog) is str and catalog in ["default", "gaia_calibration"]:
            catalog = "gaia_calibration"
            if catalog not in self.catalog:
                self.fetch_calibration_catalog(name=catalog, **fitcatprop)

        # 1.
        # - Create the piff stars
        stars, (fitcat, inputfile) = self.get_stars(catalog, fileout="default",
                                                    filtered=on_filtered_cat,
                                                    update_config=True,
                                                    fullreturn=True)
        # - and record the information
        self._fitcatalog = fitcat
        self.config['calibration_cat'] = self.fitcatalog.get_config()

        # 2.
        # - Build the PSF object 
        psf = piff.SimplePSF.process(self.config['psf'])
        wcs, pointing = self.get_wcspointing(inputfile=inputfile)
        # - and fit the PSF
        psf.fit(stars, wcs, pointing, logger=self.logger)
        
        # 3.
        # - Store the results
        [psf.write(p + save_suffix) for p in self.get_prefix()]
        [self.save_config(p + 'piff_config.json') for p in self.get_prefix()]
        # - save on the current object.
        self.set_psf(psf)

    # -------- #
    # BUILDER  #
    # -------- #
    def fetch_calibration_catalog(self, name='gaia_calibration',
                                       gmag_range=[14,16],
                                       rpmag_range=None,
                                       bpmag_range=None,
                                       colormag_range=None,
                                       bound_padding=30, 
                                       seplimit=8, **kwargs):
        """ """
        self.fetch_gaia_catalog(gmag_range=gmag_range,
                                rpmag_range=rpmag_range,bpmag_range=bpmag_range,
                                colormag_range=colormag_range,
                                bound_padding=bound_padding,
                                name=name,
                                seplimit=seplimit, **kwargs
                                )

    def fetch_comparison_catalog(self, name='gaia_comp',
                                       gmag_range=[14,20],
                                       rpmag_range=None,
                                       bpmag_range=None,
                                       colormag_range=None,
                                       bound_padding=30, 
                                       seplimit=3, **kwargs):
        """ """
        self.fetch_gaia_catalog(gmag_range=gmag_range,
                                rpmag_range=rpmag_range,bpmag_range=bpmag_range,
                                colormag_range=colormag_range,
                                bound_padding=bound_padding,
                                name=name,
                                seplimit=seplimit, **kwargs
                                )

    
    # -------- #
    #  GETTER  #
    # -------- #
    def fetch_gaia_catalog(self, gmag_range=[14,20],
                            rpmag_range=None,
                             bpmag_range=None,
                             colormag_range=None,
                             bound_padding=30,
                             isolated=True, 
                             name='gaia',
                             seplimit=8, setit=True,
                             setsky=True, setwcs=True, setmask=True,
                             as_collection=True):
        """ """
        # To be changed using fetch_catalog
        cat_ = self.fetch_catalog("gaia", setit=False,
                                setsky=setsky, setwcs=setwcs, setmask=setmask,
                                add_boundfilter=True, # ignored if bound_padding is None
                                bound_padding=bound_padding)
        cat_.change_name(name)
        cat_.data['ra'] = cat_.data['RA_ICRS']
        cat_.data['dec'] = cat_.data['DE_ICRS']

        # Work the same for single or non-single Catalog
        # Filters
        if setmask:
            cat_.add_filter('masked', False, name='maskedout')
        
        if gmag_range is not None:
            cat_.add_filter('Gmag', gmag_range, name='gmag_outrange')
            
        if colormag_range is not None:
            cat_.add_filter('RPmag', rpmag_range, name='rpmag_outrange')
            
        if bpmag_range is not None:
            cat_.add_filter('BPmag', gpmag_range, name='bpmag_outrange')
            
        if isolated and seplimit is not None:
            cat_.measure_isolation(seplimit=seplimit)
            cat_.add_filter('is_isolated', True, name='not_isolated')
            
        if setit:
            self.set_catalog(cat_, name=name)
            
        else:
            return c if as_collection else c.catalogs

    
    def compute_shapes(self, stars, save=False, save_suffix = 'shapes'):
        """ """
        shapes = {'instru_flux': [], 'T_data': [], 'T_model': [],
                      'g1_data': [],'g2_data': [],'g1_model': [],
                      'g2_model': [],'u': [],'v': [],
                      'flag_data': [],'flag_model': [],
                      'center_u' : [],'center_v' : []}
        for s in stars:
            s.run_hsm()
            ns = self.psf.drawStar(s)
            ns.run_hsm()
            shapes['instru_flux'].append(s.flux)
            shapes['T_data'].append(s.hsm[3])
            shapes['T_model'].append(ns.hsm[3])
            shapes['g1_data'].append(s.hsm[4])
            shapes['g1_model'].append(ns.hsm[4])
            shapes['g2_data'].append(s.hsm[5])
            shapes['g2_model'].append(ns.hsm[5])
            shapes['u'].append(s.u)
            shapes['v'].append(s.v)
            shapes['center_u'].append(s.center[0])
            shapes['center_v'].append(s.center[1])
            shapes['flag_data'].append(s.hsm[6])
            shapes['flag_model'].append(ns.hsm[6])
            
        shapes['T_data_normalized'] = shapes['T_data']/np.median(shapes['T_data'])
        shapes['T_model_normalized'] = shapes['T_model']/np.median(shapes['T_data'])
        # Adding cat_kwargs
        shapes = {**shapes, **get_stars_cat_kwargs(stars)}
        if save:
            [np.savez(p + save_suffix,**shapes) for p in self.get_prefix()]
            
        return shapes

    def compute_residuals(self, stars, normed = True, sky = 100):
        """ """
        residuals = []
        for s in stars:
            draw = self.psf.drawStar(s)
            res = s.image.array - draw.image.array
            if normed:
                res /= draw.image.array + sky
            residuals.append(res)
        return np.stack(residuals)


    # ================ #
    #   Properties     #
    # ================ #
    # - Catalogs
    @property
    def fitcatalog(self):
        """ """
        if not hasattr(self, "_fitcatalog"):
            return None
        return self._fitcatalog

    
######################
#                    #
#  Ziff Collection   #
#                    #
######################

class ZiffCollection( object ):
    
    def __init__(self, sciimg_list, mskimg_list = None, logger = None, **kwargs):
        """ 
        
        Parameters
        ----------
        sciimg_list, mskimg_list: [strings or list of] -optional-
            Path (or list of) to the ztf science image (sciimg.fits) and their corresponding mask images (mskimg.fits)

        logger: [logger or None] -optional-
            logger passed to piff.

        **kwargs goes to Ziff
        """
        if mskimg_list is None:
            mskimg_list = [None] * len(sciimg_list)
            
        self.ziffs = [Ziff(s,m,logger,**kwargs) for (s,m) in zip(np.atleast_1d(sciimg_list), np.atleast_1d(mskimg_list))]
    
    @classmethod
    def from_zquery(cls, zquery,  groupby = ['ccdid','fracday','fid'], **kwargs):
        """ """
        mt = zquery.get_local_metatable(which='dl')
        mt.index = np.arange(np.size(mt,axis=0))
        groupby = mt.groupby(groupby)
        groups = groupby.groups
        local_data_sciimg = np.asarray(zquery.get_local_data("sciimg.fits", filecheck = False))
        sciimg_list = [local_data_sciimg[groupby.get_group(i).index.values] for i in groups]
        local_data_mskimg = np.asarray(zquery.get_local_data("mskimg.fits", filecheck = False))
        mskimg_list = [local_data_mskimg[groupby.get_group(i).index.values] for i in groups]
        return cls(sciimg_list, mskimg_list, **kwargs) #cls(name, date.today().year - year)

    def to_file(self, filename):
        with open(filename,'w') as f:
            for ziff in self.ziffs:
                for (i,l0) in enumerate(ziff._sciimg):
                    if i ==0 :
                        f.write(l0)
                    else:
                        f.write(',' + l0)
                f.write('\n')
                
    @classmethod
    def from_file(cls, filename,max_rows=None, **kwargs):
        list_img = []
        with open(filename,'r') as f: 
            lines = f.readlines()
            if max_rows is None:
                max_rows = len(lines)
            for line in lines[0:max_rows]: 
                list_img.append(line[0:-1].split(',')) 
        return cls(list_img, **kwargs)

    def read_shapes(self):
        dfs = []
        for (i,z) in enumerate(self.ziffs):
            print('{i+1}/{len(self.ziffs)}')
            try:
                df = z.read_shapes()
                df['ccd'] = z.ccdid[0]
                df['fracday'] = z.fracday[0]
                df['quadrant'] = z.qid[0]
                df['MAGZP'] = z.get_header()[0]['MAGZP']
                df['filter'] = z.filter[0]
                dfs.append(df)
            except FileNotFoundError:
                print(f"ziff {i+1} not found")
        return pd.concat(dfs)
    

    def eval_func(self, attr, parallel = False, **kwargs):
        return [getattr(z,attr)(**kwargs) for z in self.ziffs]

    def eval_func_stars(self,attr, stars_list, parallel = False, **kwargs):
        return [getattr(z,attr)(stars = stars_list[i],**kwargs) for (i,z) in enumerate(self.ziffs)]

# End of ziff.py ========================================================
