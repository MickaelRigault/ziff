#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Base ZIFF Classes """

import os
import numpy as np
import logging
import json

# PIFF 
import piff

# ZTFImage
from ztfimg import image as ztfimage

from . import catalog
######################
#                    #
#  LOG & CONFIG      #
#                    #
######################

class _ZIFFLogConfig_( object ):
    # CONFIG & LOGGER
    # ================ #
    #   Methods        #
    # ================ #
    # -------- #
    #  I/O     #
    # -------- #
    def save_config(self, path):
        """ """
        with open(path, 'w') as f:
            json.dump(self.config, f)

    # -------- #
    #  LOADER  #
    # -------- #
    def load_default_config(self):
        """ Load the default configuration settings using default configuration file """
        import pkg_resources
        file_name = pkg_resources.resource_filename('ziff', 'data/default_config.json')
        with open(file_name) as config_file:
            self.set_config( json.load(config_file) )

    # -------- #
    #  SETTER  #
    # -------- #
    def set_logger(self, logger=None, basename=None):
        """ set the logger that is going to be passed to piff """
        if logger is None:
            if basename is None:
                basename = ""
            logging.basicConfig(filename= basename + 'logger.piff',level=logging.DEBUG)
            logger = logging.getLogger()
            self._logger = logger
        else:
            self._logger = logger

    def set_config(self, config=None):
        """ """
        if config is None or config in ["def","default"]:
            self.load_default_config()
        else:
            self._config = config

    # - Update configuration
    def set_config_value(self, key_path, value, sep=','):
        """ update the configuration values """
        # Should be cleaned but it works
        kp = key_path.split(sep)
        to_eval = 'config' + ''.join([f"['{k}']" for k in kp]) + f" = {value}"
        if isinstance(value, str):
            to_eval = 'config' + ''.join([f"['{k}']" for k in kp]) + f" = '{value}'"
        exec(to_eval,{'config':self.config})

    # -------- #
    #  GETTER  #
    # -------- #
    def get_piff_inputfile(self, catfile=None):
        """ get the PIFF input file given your logger and configurations """
        if catfile is not None:
            self.config['i/o']['cat_file_name'] = list(np.atleast_1d(catfile))
            
        inputfile = piff.InputFiles(self.config['i/o'], logger=self.logger)
        inputfile.setPointing('RA','DEC')
        return inputfile

    def get_wcspointing(self):
        """ """
        inputfile = self.get_piff_inputfile()
        wcs = inputfile.getWCS()
        inputfile.setPointing('RA','DEC')
        return wcs,inputfile.getPointing()

    def get_config_value(self, key, squeeze=True):
        """ get a config value (value.s or dict) 

        Parameters
        ----------
        key: [string]
            name of the key you are looking for.
            with config = {conf1:{a1:vala1, b1:valb1,..},
                         conf2:{a2:vala2, b2:valb2,..}
                         }
            if key is e.g. conf1, this returns the full dict
            if key is e.g. a2, this returns vala2.

        squeeze: [bool] -optional-
            if True and a single value returned, value[0] given and None if no Value
            if False [value] returned

        Returns
        -------
        list (or None/value, see squeeze) or dict depending on the given key.
        """
        if key in self.config.keys():
            return self.config[key]
        
        value = [v_ 
                     for name_, config_ in self.config.items() 
                     for k_,v_ in config_.items() 
                     if k_ == key]
        if squeeze:
            if len(value)==1:
                return value[0]
            if len(value)==0:
                return None
        return value        
    # ================ #
    #   Properties     #
    # ================ #
    @property
    def logger(self):
        """ """
        if not hasattr(self, "_logger"):
            return self.set_logger(None)
        return self._logger

    @property
    def config(self):
        """ """
        if not hasattr(self, "_config"):
            self.set_config("default")
        return self._config
            
    
######################
#                    #
#     IMAGES         #
#                    #
######################

class _ZIFFImageHolder_( _ZIFFLogConfig_ ):
    # IMAGES & MASK (on top of logger & config)
    
    @classmethod
    def from_ztfimage(cls, ztfimage, logger=None, **kwargs):
        """ """
        this = cls(**kwargs)
        this.set_logger(logger)
        this.set_images(ztfimage)
        return this

    @classmethod
    def from_zquery(cls, zquery, logger=None, **kwargs):
        """ """
        sciimg_list = zquery.get_local_data("sciimg.fits")
        mskimg_list =  zquery.get_local_data("mskimg.fits")
        this = cls(**kwargs)
        this.set_logger(logger)
        this.load_images(sciimg_list, mskimg_list)
        return this
    
    # ================ #
    #   Methods        #
    # ================ #
    def load_images(self, imagefile, maskfile=None, download=False):
        """ Builds the ztfimages from the given filepath and calls self.set_images() """
        from ztfimg import image
        
        # Handles list / single
        imagefile = np.atleast_1d(imagefile)
        
        # Handles mask I/O
        if maskfile is not None:
            maskfile = np.atleast_1d(maskfile)
            if len(maskfile) != len(imagefile):
                raise ValueError("You gave {len(maskfile)} masks and {len(imagefile)} images. size of mskimg must match or be None")
        else:
            maskfile = [None for i in range(len(imagefile))]

        # Build the ztfimage
        zimages = [ztfimage.ScienceImage.from_filename(image_, filenamemask=mask_, download=download)
                         for (image_, mask_) in zip(imagefile, maskfile)]
        
        self.set_images(zimages)
        
    # -------- #
    #  SETTER  #
    # -------- #
    def set_images(self, ztfimages):
        """ Set a (list of) image(s) to the Ziff instance. """ 
        ztfimg =  np.atleast_1d(ztfimages)
        for ztfimg_ in ztfimg:
            if ztfimage.ZTFImage not in ztfimg_.__class__.__mro__:
                raise TypeError("The given images must be image.ZTFImage (or inherite from) ")
            
        self._images = ztfimg
        # add the filename
        self.config['i/o']['image_file_name'] = self._sciimg
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_header(self, which=None):
        """Get the header of the image 

        Parameters
        ----------

        Returns
        -------
        """
        if which is None:
            return [img_.header for img_ in self._images]
        
        return self._images[which].header

    def get_headerkey(self, key, which=None, defaultkey=None):
        """ """
        return [h.get(key, defaultkey) for h in self.get_header(which=which)]
            
    def get_singleimage(self, num):
        """ create a new Ziff instance with single image """
        return self.__class__.from_ztfimage(self._images[num], logger=self.logger)

    def get_dir(self):
        """ Get the directory of the image """
        # IO Stuffs        
        return [os.path.dirname(s) for s in self._sciimg]

    def get_center(self, inpixel=True, **kwargs):
        """ """
        return self._read_images_property_("get_center", isfunc=True, inpixel=inpixel, **kwargs)

    def get_diagonal(self, inpixel=True, **kwargs):
        """ """
        return self._read_images_property_("get_diagonal", isfunc=True, inpixel=inpixel, **kwargs)

    
    def get_prefix(self, basename=False):
        """ """
        if basename:
            return [p_.split("/")[-1] for p_ in self.get_prefix(basename=False)]
        
        return ['_'.join(s.split('_')[0:-1])+'_' for s in self._sciimg]

    def get_mask(self, **kwargs):
        """ 
        **kwargs goes to ztfimg's image{s}.get_mask().
        => 
           from_sources=None, tracks=True, ghosts=True,
           spillage=True, spikes=True, dead=True, nan=True,
           saturated=True, brightstarhalo=True,
           lowresponsivity=True, highresponsivity=True,
           noisy=True, sexsources=False, psfsources=False,
           alltrue=False, flip_bits=True,
           verbose=False, getflags=False

        """
        return self._read_images_property_("get_mask", isfunc=True, **kwargs)

    def get_background(self, method=None, rmbkgd=False, backup_default='sep', **kwargs):
        """ Get the image{s} background using their get_background() method.

        Parameters
        ----------
        method: [string] -optional-
            if None, method ="default"
            - "default": returns the background store as self.background (see set_background)
            - "median": gets the median of the fully masked data (self.get_mask(alltrue=True))
            - "sep": returns the sep estimation of the background image (Sextractor-like)

        rmbkgd: [bool] -optional-
            // ignored if method != median //
            shall the median background estimation be made on default-background subtraced image ?

        backup_default: [string] -optional-
            If no background has been set yet, which method should be the default backgroud.
            If no background set and backup_default is None an AttributeError is raised.

        **kwargs goes to image{s}.get_background()
        Returns
        -------
        float/array (see method)
        """
        return self._read_images_property_("get_background", isfunc=True,
                                            method=method, rmbkgd=rmbkgd, backup_default=backup_default, **kwargs)
    # ================ #
    #   Properties     #
    # ================ #
    # - Images
    @property
    def images(self):
        """ ztfimg.ZTFImage (or child of) """
        if not hasattr(self, "_images"):
            return None
        
        return self._images if not self.is_single() else self._images[0]

    def has_images(self):
        """ test if there are any images attached to this instance. """
        return hasattr(self,"_images") and self._images is not None and len(np.atleast_1d(self._images))>0

    @property
    def nimgs(self):
        """ Number of images used"""
        if self.has_images():
            return len(self._images)
        return None
    
    def is_single(self):
        """ """
        return self.nimgs==1

    def _read_images_property_(self, key, isfunc=False, *args, **kwargs):
        """ """
        if not isfunc:
            return getattr(self.images,key) if self.is_single() else \
              [getattr(img_,key) for img_ in self.images]
              
        return getattr(self.images,key)(*args, **kwargs) if self.is_single() else \
          [getattr(img_,key)(*args, **kwargs) for img_ in self.images]
    

    @property
    def _sciimg(self):
        """ fullpath of the sciimg """
        if self.has_images():
            return [img_._filename for img_ in self._images]
        return None
    
    @property
    def mask(self):
        """ """
        return self.get_mask()

    @property
    def wcs(self):
        """ """
        return self._read_images_property_("wcs")

    @property
    def header(self):
        """ shortcut to get_header. Squeezed """
        return self._read_images_property_("header")
    
    @property
    def ccdid(self):
        """ ccd ID of the loaded images """
        return self._read_images_property_("ccdid")

    @property
    def qid(self):
        """ quadran ID of the loaded images """
        return self._read_images_property_("qid")

    @property
    def rcid(self):
        """ RC ID of the loaded images (quandrant ID and ccd ID) """
        return self._read_images_property_("rcid")

    @property
    def prefix(self):
        """ Prefix of the image. Useful for getting mskimg, psfcat etc. 
        in the actual pipeline. Squeezed"""
        return self.get_prefix()[0] if self.is_single() else self.get_prefix()

    @property
    def fracday(self):
        """ """
        if self.is_single():
            return int(self.prefix.split('/')[-2][1::])
        return [int(p.split('/')[-2][1::]) for p in self.prefix]

    @property
    def filter(self):
        """ """
        if self.is_single():
            return self.prefix.split('_')[-5][1::]
        return [p.split('_')[-5][1::] for p in self.prefix]

    @property 
    def shape(self):
        """ ZTF quadrant umage shape (3080, 3072)"""
        return (3080, 3072)


class ZIFF( _ZIFFImageHolder_, catalog._CatalogHolder_  ):
    
    def __init__ (self, sciimg=None, mskimg=None,
                      logger=None, catalog=None,
                      config="default", download=True):
        """Wrapper of PIFF for ZTF 

        Single fit of potentially multi images.
        
        Parameters
        ----------
        sciimg: [string]
            path to the ztf science image (sciimg)
            
        mskimg: [string or None] -optional-
            path to the science image mask (mskimg)

        logger: [logger or None] -optional-
            logger passed to piff. 
        
        """
        # Must start with config.
        # - Config
        self.set_config(config)

        # - Data
        if sciimg is not None:
            self.load_images(sciimg, mskimg, download=download)

        # - Catalog
        if catalog is not None:
            self.set_catalog(catalog, name="calibrator")
            
            
        # - Logger            
        if logger is not None:
            self.set_logger(logger)

    @classmethod
    def from_file(cls, filename, row=0, **kwargs):
        """ """
        with open(filename,'r') as f: 
            lines = f.readlines()
            for (i,line) in enumerate(lines):
                if i == row:
                    sciimg = line[0:-1].split(',')
                    return cls(sciimg, **kwargs)
    # ================ #
    #   Methods        #
    # ================ #
    # ------- #
    #  SETTER #
    # ------- #
    def set_psf(self, psf):
        """ """
        self._psf = psf

    def fetch_catalog(self, which="gaia", name=None, setit=True,
                          setsky=True, setwcs=True, setmask=True,
                          add_boundfilter=True, bound_padding=30,
                          **kwargs):
        """ **kwargs goes to catalog.fetch_ziff_catalog() """
        
        catalog_ = catalog.fetch_ziff_catalog(self, which=which, as_collection=True, **kwargs)
        
        if catalog_.name is None:
            catalog_.change_name(which)

        if setwcs:
            catalog_.set_wcs(self.wcs)
            
        if setsky:
            sky = self.get_background()
            stampsize = self.get_config_value("stamp_size")
            catalog_.build_sky_from_bkgdimg(sky, stampsize)

        if setmask:
            mask = self.get_mask()
            stampsize = self.get_config_value("stamp_size")
            catalog_.build_mask_from_maskimg(mask, stampsize)
            
        if add_boundfilter and bound_padding is not None:
            ymax, xmax = self.shape
            catalog_.add_filter('xpos',[bound_padding, xmax-bound_padding],
                                    name = 'xpos_out')
            catalog_.add_filter('ypos',[bound_padding, ymax-bound_padding],
                                    name = 'ypos_out')

        if not setit:
            return catalog_
        
        self.set_catalog(catalog_, name=name)
    # ------- #
    # PIFF    #
    # ------- #
    def catalog_to_stars(self, catalog, fileout=None, append_df_keys = None, **kwargs):
        """ """
        cat, catfile = self._get_stored_catalog_(catalog, fileout, **kwargs)
        inputfile = self.get_piff_inputfile(catfile=catfile)
        stars = inputfile.makeStars(logger=self.logger)
        
        for s in stars:
            s._cat_kwargs = {}
            
        if append_df_keys is not None:
            append_df_keys = np.atleast_1d(append_df_keys)
            df = self.get_stacked_cat_df()[catalog]
            for (i,s) in enumerate(stars):
                for key in append_df_keys:
                    s._cat_kwargs[key] = df.iloc[i][key]
                    s._cat_kwargs['name'] = df.iloc[i].name
        return stars

    def reflux_stars(self, stars, fit_center = False, which = 'piff', show_progress=True):
        """ measure the flux and centroid (if allowed) of the star give the PSF.
        

        Parameters
        ----------
        stars: [piff.Stars (list of)]
        
        fit_center: [bool] -optional-


        which: [string] -optional-
            how to fit for the reflux?
        
        show_progress: [bool] -optional-
            Show progress bar (astropy.utils.console.ProgressBar)

        Returns
        -------
        piff.Stars
        """
        if fit_center:
            self.psf.model._centered = True
            
        wcs, pointing = self.get_wcspointing()
        new_stars = []
        if show_progress:
            from astropy.utils.console import ProgressBar
            from .utils import is_running_from_notebook
            bar = ProgressBar( len(stars), ipython_widget=is_running_from_notebook() )
        else:
            bar = None
            
        for (i,s) in enumerate(stars):
            if bar is not None:
                bar.update(i)
                
            s.image.wcs = wcs[s.chipnum]
            s.run_hsm()
            new_s = self.psf.model.initialize(s)
            new_s = self.psf.interpolateStar(new_s)
            new_s.fit.flux = s.run_hsm()[0]
            new_s.fit.center = (0,0)
            if which == 'minuit':
                new_s = self.reflux_minuit(new_s, fit_center = fit_center)
            else:
                new_s = self.psf.model.reflux(new_s, fit_center = fit_center)
                
            new_s._cat_kwargs = s._cat_kwargs
            new_stars.append(new_s)

        if bar is not None:
            bar.update( len(stars) )

        self.psf.model._centered = False
        return new_stars


    @property
    def psf(self):
        """ """
        if not self.has_psf():
            return None
        return self._psf

    def has_psf(self):
        """ """
        return hasattr(self,"_psf") and self._psf is not None
