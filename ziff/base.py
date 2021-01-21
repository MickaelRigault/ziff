#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Base ZIFF Classes """

import os
import numpy as np
import logging
import json
import warnings
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

    def set_stampsize(self, stampsize):
        """ Size of the stamps used for the PSFF psf fit """
        self.set_config_value('i/o,stamp_size', int(stampsize))

    def set_nstars(self, nstars):
        """ Number of stars used for the PIFF psf fit. """
        self.set_config_value('i/o,nstars', int(nstars))

    # -------- #
    #  GETTER  #
    # -------- #
    def get_piff_inputfile(self, catfile=None, ioconfig=None, verbose=False):
        """ get the PIFF input file given your logger and configurations 
        
        ioconfig: [dict]  -optional-
            dictionary containing the input information for piff.
            as in self.config['i/o']
        
        """
        if ioconfig is None:
            ioconfig = self.config['i/o']
            
        if catfile is not None:
            ioconfig['cat_file_name'] = list(np.atleast_1d(catfile))
            
        if verbose:
            print(ioconfig)
            
        inputfile = piff.InputFiles(ioconfig, logger=self.logger)
        inputfile.setPointing('RA','DEC')
        return inputfile

    def get_wcspointing(self, inputfile=None, verbose=False):
        """ """
        if inputfile is None:
            inputfile = self.get_piff_inputfile(verbose=verbose)
            
        wcs = inputfile.getWCS()
        inputfile.setPointing('RA','DEC')
        return wcs, inputfile.getPointing()

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

    def get_imagedata(self, which="data", **kwargs):
        """ high level method for accessing data, background or mask images.
        See also: get_data(), get_background(), get_mask() for additional tools.

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
            

        **kwargs goes to the method used.
        Returns
        -------
        2d image(s)
        """
        # data
        if which in ["data"]:
            return self.get_data(applymask=True, rmbkgd=True, **kwargs)

        if which in ["raw"]:
            return self.get_data(applymask=False, rmbkgd=False, **kwargs)

        if which in ["rawmasked"]:
            return self.get_data(applymask=True, rmbkgd=False, **kwargs)
        
        # background
        if which in ["background", "bkgd"]:
            return self.get_background(**kwargs)
        
        # mask
        if which in ["mask", "bkgd"]:
            return self.get_mask(**kwargs)

    def get_imageprop(self, key, **kwargs):
        """ gets [img.`key` for img in images] """
        return self._read_images_property_(key, isfunc=False, **kwargs)
    
        
    def get_data(self, applymask=True, maskvalue=np.NaN, rmbkgd=True, whichbkgd="default"):
        """ """
        return self._read_images_property_("get_data", isfunc=True,
                                               applymask=applymask,
                                            maskvalue=maskvalue, rmbkgd=rmbkgd, whichbkgd=whichbkgd)
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
    
    def __init__(self, sciimg=None, mskimg=None, psffile=None,
                      logger=None, catalog=None,
                      config="default", fetch_psf=False,
                      download=True):
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

        # - PSF file
        if psffile is not None or fetch_psf:
            self.load_psf(psffile, fetch=fetch_psf)

    @classmethod
    def from_file(cls, filename, row=0, **kwargs):
        """ """
        with open(filename,'r') as f: 
            lines = f.readlines()
            for (i,line) in enumerate(lines):
                if i == row:
                    sciimg = line[0:-1].split(',')
                    return cls(sciimg, **kwargs)

    def store_psfshape(self, catalog, psf=None, which=['stars', 'psfmodel'], filtered=True,
                           add_imgprop=True, add_filter=None, fileout=None, getshape=False):
        """ """
        data = self.get_psfshape(catalog, psf=psf, which=which, filtered=filtered,
                                     add_imgprop=add_imgprop, add_filter=add_filter)
        if self.is_single():
            if fileout is None:
                fileout = self.prefix+'psfshape.csv' 
            data.to_csv(fileout)
        else:
            raise NotImplementedError("No shape datastorage implemented for non-single ziff.")

        if getshape:
            return data

    # ================ #
    #   Methods        #
    # ================ #
    # ------- #
    # LOADER  #
    # ------- #
    def load_psf(self, psffilename=None, fetch=True):
        """ loads a piff output file: `bla`_output.piff'
        This contains the PSF properties
        """
        if psffilename is None and fetch:
            from ztfquery import buildurl
            psffilename_ = [buildurl.filename_to_scienceurl(prefix_, source="local", suffix="output.piff", check_suffix=False)
                               for prefix_ in self.get_prefix()]
            psffilename = [None if not os.path.isfile(psf_) else psf_ for psf_ in psffilename_]
            if self.is_single():
                psffilename = psffilename[0]

                
        # TO BE TESTED, CASE WITH MULTI IMAGES.
        if psffilename is not None:
            self.set_psf( piff.PSF.read(file_name=psffilename, logger=self.logger) )
            # tmp patch:
            #self.psf.wcs = list(np.atleast_1d(self.psf.wcs))        
        
    # ------- #
    # SETTER  #
    # ------- #
    def set_psf(self, psf):
        """ """
        self._psf = psf

    def fetch_catalog(self, which="gaia", name=None, setit=True,
                          setsky=True, setwcs=True, setmask=True,
                          add_boundfilter=True, bound_padding=50,
                          isolationlimit=None,
                          **kwargs):
        """ **kwargs goes to catalog.fetch_ziff_catalog() """
        print("fetch_catalog IS DEPRECATED")
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
            catalog_.add_filter('masked', False, name='maskedout')
            
        if add_boundfilter and bound_padding is not None:
            ymax, xmax = self.shape
            catalog_.add_filter('xpos',[bound_padding, xmax-bound_padding],
                                    name = 'xpos_out')
            catalog_.add_filter('ypos',[bound_padding, ymax-bound_padding],
                                    name = 'ypos_out')

        if isolationlimit is not None:
            catalog_.measure_isolation(seplimit=isolationlimit)
            catalog_.add_filter('is_isolated', True, name='not_isolated')
            
            
        if not setit:
            return catalog_
        
        self.set_catalog(catalog_, name=name)

    def fetch_ps1cal_catalog(self, setit=True,
                                 name="ps1cal",
                                 setsky=True, setwcs=True, setmask=True,
                                 add_boundfilter=True, bound_padding=50,
                                 isolationlimit=None,
                                 gmag_range=None, rmag_range=None,
                                 imag_range=None, zmag_range=None,
                                 ):
        """ """
        dataframes = self._read_images_property_("get_ps1_calibrators", isfunc=True)
        if self.is_single():
            ps1cat = catalog.Catalog(dataframes.rename(columns={"x":"xpos","y":"ypos"}),
                                  name="ps1cal")
        else:
            catlist = [catalog.Catalog(df_.rename(columns={"x":"xpos","y":"ypos"}), name=name_)
                         for i,(df_,name_) in enumerate(zip(dataframes, self.get_prefix(True))) ]
            ps1cat = catalog.CatalogCollection(catlist, load_data=True)

        catalog_ = self._enrich_cat_(ps1cat,
                                    name=name,
                                    setsky=setsky, setwcs=setwcs, setmask=setmask,
                                    add_boundfilter=add_boundfilter, bound_padding=bound_padding,
                                    isolationlimit=isolationlimit)
        if gmag_range is not None:
            catalog_.add_filter('gmag', gmag_range, name='gmag_outrange')
        if rmag_range is not None:
            catalog_.add_filter('rmag', rmag_range, name='rmag_outrange')
        if imag_range is not None:
            catalog_.add_filter('imag', imag_range, name='imag_outrange')
        if zmag_range is not None:
            catalog_.add_filter('zmag', zmag_range, name='zmag_outrange')

        if not setit:
            return catalog_

        self.set_catalog(catalog_, name=name)

    def _enrich_cat_(self, catalog_, name=None,
                         setsky=True, setwcs=True, setmask=True,
                         add_boundfilter=True, bound_padding=50,
                         isolationlimit=None):
        """ """
        if catalog_.name is None:
            catalog_.change_name(name)

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
            catalog_.add_filter('masked', False, name='maskedout')
            
        if add_boundfilter and bound_padding is not None:
            ymax, xmax = self.shape
            catalog_.add_filter('xpos',[bound_padding, xmax-bound_padding],
                                    name = 'xpos_out')
            catalog_.add_filter('ypos',[bound_padding, ymax-bound_padding],
                                    name = 'ypos_out')

        if isolationlimit is not None:
            catalog_.measure_isolation(seplimit=isolationlimit)
            catalog_.add_filter('is_isolated', True, name='not_isolated')

        return catalog_
    # ------- #
    # GETTER  #
    # ------- #
    def eval_psf(self, xpos, ypos, chipnum=0, flux=1.0, offset=(0, 0),
                    stamp_size=None, image=None, informat="numpy",
                    asarray=True,**kwargs):
        """ """
        if not self.has_psf():
            raise AttributeError("No PSF loaded.")
        
        if stamp_size is None:
            stamp_size = self.get_config_value("stamp_size")

        # not really important as the PSF is not varying that fast.
        formatoffset = -1 if informat == "numpy" else 0
            
        galsimimg = self.psf.draw(xpos-formatoffset, ypos-formatoffset,
                                      stamp_size=stamp_size,
                                      chipnum=chipnum,
                                      flux=flux, offset=offset, image=image)
        if asarray:
            return galsimimg.array
        
        return galsimimg

    def get_psf(self, catalog, chipnum=0, flux=1.0, iloc=None, **kwargs):
        """ """
        cat = self.get_catalog(catalog, chipnum=chipnum)
        xpos, ypos = cat.get_data()[["xpos","ypos"]] if iloc is None else cat.get_data().iloc[iloc][["xpos","ypos"]]
        return [eval_psf(xpos_, ypos_, chipnum=chipnum, flux=flux, **kwargs)
                    for xpos_, ypos_ in zip(xpos, ypos) ]

    def get_psfshape(self, catalog, psf=None, which=["stars", "psfmodel"], filtered=True, add_imgprop=True,
                         add_filter=None, verbose=False):
        """ """
        import pandas
        from . import star
        stars, (cat, inputfile) = self.get_stars( catalog, filtered=filtered,
                                                   fullreturn=True, add_filter=add_filter,
                                                      verbose=verbose)
        soll = star.StarCollection(stars)
        if "psfmodel" in which:
            if psf is None:
                psf = self.psf
            soll.measure_psfmodel(psf)
            
        soll.measure_shapes(which)

        # Building the returned dataframe
        catmag = cat.data
        stars  = soll.shapes["stars"].set_index(catmag.index)
        model  = soll.shapes["psfmodel"].set_index(catmag.index)
        
        mshapes = pandas.merge(stars, model, left_index=True, right_index=True, suffixes=("_stars","_model"))
        dataout = pandas.merge(catmag, mshapes, left_index=True, right_index=True)
        if add_imgprop:
            if not self.is_single():
                print("add_imgprop not implemented yet for non-single images")
            else:
                for key in ["ccdid","rcid","qid","fieldid","filterid","exptime","obsjd"]:
                    dataout[key] = self.get_imageprop(key)
                    
        return dataout
    
    def get_stamp(self, catalog, which="data", filtered=True, fullreturn=False, **kwargs):
        """ 
        Parameters
        ----------
        catalog: [string/catalog]
            catalog to be used to fetch xpos and ypos.
        
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
        return self._get_single_stamp_(catalog, which=which, filtered=filtered, fullreturn=fullreturn, **kwargs)
            
    def _get_single_stamp_(self, catalog, which="data", filtered=True, xyformat="numpy", fullreturn=False, **kwargs):
        """ """
        cat = self.get_catalog(catalog)
        stamps = cat.get_datastamps(self.get_imagedata(which=which, **kwargs),
                                      self.get_config_value("stamp_size"), filtered=filtered,
                                      xyformat=xyformat)
        if fullreturn:
            return stamps, cat.get_data(filtered=filtered)
        return stamps

    def get_stars(self, catalog, fileout="tmp", fullreturn=False,
                      update_config=False, nstars="no_limit",
                      filtered=True, verbose=False, **kwargs):
        """ return PIFF stars for the given catalog using get_piff_inputfile().makeStars() 

        Parameters
        ----------

        fileout: [string or None] -optional-
            how should the catalog file be stored to be passed to piff ?
            - 'tmp': temporary name: tmp_`bla`
            - None or 'default': using the cat.build_filename(prefix) method
            - rest: considered as the actuel filename.

        fullreturn: [bool] -optional-
            returns the catalog and the piff inputfile in addition to the stars 
            (stars, cat)

        Returns
        -------
        piff.Stars or piff.Stars, (catalog, inputfile)
        """
        # 1.
        # - parse catalog
        cat, catfile = self._get_stored_catalog_(catalog, fileout=fileout, filtered=filtered,
                                                **{**{"xyformat":"fortran"},**kwargs})
        if cat.npoints == 0:
            warnings.warn("No entry in the given catalog, no stars to get.")
            if not fullreturn:
                return None
            return None, (cat, None)
        
        # 2.
        # - build the piff input file using a copy of the i/o config
        if not update_config:
            ioconfig = self.get_config_value("i/o").copy()
        else:
            ioconfig = self.get_config_value("i/o")

        #
        if nstars is not None:
            if type(nstars) is str:
                if nstars in ["no_limit"]:
                    ioconfig["nstars"] = len(cat.data)+1
                else:
                    raise ValueError("Cannot parse given nstars")
            else:
                ioconfig["nstars"] = nstars
            
        ioconfig['cat_file_name'] = list(np.atleast_1d(catfile))
        inputfile = self.get_piff_inputfile(ioconfig=ioconfig, verbose=verbose)

        # 3.
        # - build stars from the inputfil
        stars = inputfile.makeStars(logger=self.logger)
        
        if not fullreturn:
            return stars

        return stars, (cat, inputfile)

    # ------- #
    # FITTER  #
    # ------- #
    def get_star_psfmodel(self, stars, normed=False, asarray=False):
        """
        Parameters
        ----------
        
        Returns
        -------
        2d array or piff.Star
        """
        #
        # - Multiple Case, simply loop.
        if len(np.atleast_1d(stars))>1:
            return [self.get_star_psfmodel(star_) for star_ in stars]

        #
        # - Single Case
        star = np.atleast_1d(stars)[0]
        if not normed:
            target_star = self.psf.interpolateStar(self.psf.model.initialize( star ))
            new_star = self.psf.model.draw( target_star )
        else:
            new_star = self.psf.drawStar( star )
            # new_star.fit.params
            
        return new_star.image.array if asarray else new_star

    def get_psfmodel(self, catalog, filtered=True, normed=False, verbose=False):
        """ """
        stars = self.get_stars(catalog, fileout="tmp",
                               filtered=filtered, update_config=False,
                               fullreturn=False, verbose=verbose)
        
        return self.get_star_psfmodel(stars, normed=normed)
    
    def get_refluxed(self, catalog, filtered=True, which="piff", fit_center=False, show_progress=True, verbose=False):
        """ """
        print("Not clear if this should be used.")
        stars, (fitcat, inputfile) = self.get_stars(catalog, fileout="tmp",
                                                    filtered=filtered,
                                                    update_config=False,
                                                    fullreturn=True, verbose=verbose)
        
        wcs, pointing = self.get_wcspointing(inputfile=inputfile)
            
        new_stars = []
        for (i,s) in enumerate(stars):
            s.image.wcs = wcs[s.chipnum]
            s.run_hsm()
            new_s = self.psf.interpolateStar(self.psf.model.initialize( s ))
            new_s.fit.flux = s.run_hsm()[0]
            new_s.fit.center = (0,0)
            new_s = self.psf.model.reflux(new_s, fit_center = fit_center)    
            new_stars.append(new_s)

        return new_stars, stars
    
    # ------- #
    # PLOTTER #
    # ------- #
    def show_single_image(self, which="dataclean", chipnum=0, add_catalog=None, ax=None, zorder=3, 
               filteredcat=True, catprop={}, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        img = self.images if self.is_single() else self.images[chipnum]

        ax = img.show(which=which, ax=ax, zorder=zorder, **kwargs)
    
        if add_catalog is not None:
            cat = self.get_catalog("ps1cal", chipnum=chipnum)
            catprop_default = dict(marker="x", facecolors="C1", edgecolors="C1",)
            sc = ax.scatter(cat.get_xpos(filtered=filteredcat),
                                cat.get_ypos(filtered=filteredcat),
                                zorder=zorder+1, **{**catprop_default,**catprop},
                                )
        return ax

    def show_stamps(self, catalog, nstamps=9, ncol=3, indexes=None, which="data", filtered=True, 
                noticks=True, ssize=2.5, tight_layout=True, cmap="cividis", sort=True,
                    magkey=None, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
    
        stamps = self.get_stamp(catalog, which=which, filtered=filtered)
        if indexes is None:
            indexes = np.random.choice(np.arange(np.shape(stamps)[0]), nstamps, replace=False)
        else:
            nstamps = len(indexes)

        if sort:
            indexes = np.sort(indexes)
            
        nrow = int(np.ceil(nstamps/ncol))
        fig = mpl.figure(figsize=[ssize*ncol,ssize*nrow])
        prop = dict(origin="lower")
    
        # 
        stamps_toshow = stamps[indexes]
        cat = self.get_catalog(catalog)
        catdata = cat.get_data(filtered=filtered).iloc[indexes]
        #
        
        for i, stamp_ in enumerate(stamps_toshow):
            ax = fig.add_subplot(nrow, ncol, i+1)
            ax.imshow(stamp_, cmap=cmap, **{**prop, **kwargs})
            info = f"({catdata.iloc[i]['xpos']:.1f},{catdata.iloc[i]['ypos']:.1f})"
            if magkey is not None:
                info += f" | {catdata.iloc[i][magkey]:.1f} mag"
            info += "\n"+ f" f-out: {catdata.iloc[i]['filterout']}"
            ax.text(0.05,0.95, info, color="w", weight="bold", 
                        transform=ax.transAxes, fontsize="x-small",
                        va="top", ha="left")
            ax.set_title(indexes[i])
            if noticks:
                ax.set_xticks([])
                ax.set_yticks([])
                
        if tight_layout:
            fig.tight_layout()
            
        return fig, indexes
    
    # ------- #
    # PIFF    #
    # ------- #
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
        

    @property
    def psf(self):
        """ """
        if not self.has_psf():
            return None
        return self._psf

    def has_psf(self):
        """ """
        return hasattr(self,"_psf") and self._psf is not None
