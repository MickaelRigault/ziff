#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
from astropy.io import fits #as aio
import astropy.units as u
import os
from astropy.coordinates import Angle, SkyCoord, search_around_sky
from astropy.wcs import WCS
import copy


######################
#                    #
#     Catalog        #
#                    #
######################
class Catalog(object):
    
    def __init__(self, ziff, name):
        """ """
        self._name = name
        self._ziff = ziff
        self._filters = {}

    def __str__(self):
        """ printing method """
        out = "{} object \n".format(self.__class__.__name__)
        out += "Name  : {}\n".format(self._name)
        if hasattr(self, '_data'):
            out += "Number of stars : {}".format(np.size(self.data.loc[self.filterflag],axis=0))
        return out

    @classmethod
    def load(cls, filename, extension=None, name="catalog", ziff=None):
        """ """
        data = fits.getdata(filename, ext=extension)
        this = cls(ziff, name)
        this.set_data(pd.DataFrame(data).set_index('Source'))
        return this

    @classmethod
    def load_from_ziffztfcat(cls, ziff, name="ztfcat"):
        """ loads from ziff.ztfcat[0] """
        this = cls(ziff, name)
        f = fits.open( this.ziff.ztfcat[0] )
        this.set_data(pd.DataFrame(f[1].data).set_index('sourceid'))
        return this

    
    def copy(self, name = None, **kwargs):
        """ """        
        if name is None:
            name = self._name
            
        c = self.__class__(self._ziff, name, **kwargs)
        c._data = self._data.copy()
        c._filters = copy.deepcopy(self._filters)
        c.update_filter()
        return c
        
    # ================ #
    #   Methods        #
    # ================ #
    def change_name(self, new_name):
        """ change the name on the current catalog. """        
        self._name = new_name

    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, dataframe):
        """ Set the current dataframe as catalog data. """
        if type(dataframe) != pd.DataFrame:
            try:
                dataframe = pd.DataFrame(dataframe)
            except:
                raise TypeError("The input dataframe is not a DataFrame and cannot be converted into one.")
            
        self._data = dataframe
        for k in dataframe.keys():
            self._data[k] = self._data[k].values.byteswap().newbyteorder()
            
        self._data['filter'] = 1

    def set_dataframe(self, dataframe):
        # I WOULD REMOVE AND CALL SET_DATAFRAME -> SET_DATA
        print("DEPRECATED self.set_dataframe is deprecated use self.set_data(dataframe)")
        return self.set_data(dataframe)
    
    # - Extra Info
    def set_sky(self, sky=None):
        """ Set the sky (background) column.
        
        Parameters
        ----------
        sky: [array or None]
            values corresponding to the sky background. It will be added to the curret data as 'sky' column.
            If None, the ziff backgroud will be used.
        """
        if sky is None:
            bkgd = self.ziff.get_ztfimg()[0].get_background()
            self.data['sky'] = bkgd[np.clip(self.ypos,0,self.ziff.shape[0]-1).astype(int),np.clip(self.xpos,0,self.ziff.shape[1]-1).astype(int)]
        else:
            self.data['sky'] = sky
            
    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, filtered=False):
        """
        return_type = [df, astropy]
        """
        if not self.hasdata():
            raise AttributeError("No data set yet. Use self.set_data()")

        return self.data[self.filterflag] if filtered else self.data
        

    #  RA/Dec
    def get_skycoord(self, filtered = False):
        if filtered:
            return SkyCoord(self.ra.values[self.filtered_iindex],self.dec.values[self.filtered_iindex], unit = u.deg)
        return SkyCoord(self.ra.values,self.dec.values, unit = u.deg)
    
    def get_ra(self):
        """ Get the Right ascension column """
        keys = ['RA_ICRS','RA','ra']
        for k in keys:
            try:
                return self.data[k]
            except:
                pass
        raise ValueError("Keys {} not in dataframe".format(keys))
    
    def get_dec(self):
        """ Get the Declination ascension column """
        keys = ['DE_ICRS','DE','de','DEC','dec']
        for k in keys:
            try:
                return self.data[k]
            except:
                pass
        raise ValueError("Keys {} not in dataframe".format(keys))

    #  CCD Position (x/y)
    def get_xpos(self):
        """ Get the ccd x position column """
        # If in keys, used keys
        keys = ['x','xpos']
        for k in keys:
            try:
                return self.data[k].values
            except:
                pass
            
        # Else compute it
        return self.get_xy_from_radec()[0]
        
    def get_ypos(self):
        # If in keys, used keys
        keys = ['y','ypos']
        for k in keys:
            try:
                return self.data[k].values
            except:
                pass
        # Else compute it
        # Note that we could save them but for now we don't
        return self.get_xy_from_radec()[1]

    def get_xy_from_radec(self, update=True, overwrite=False):
        """ computes the x and y position given the ziff wcs solution and the radec values. 
        Store them as xpos and ypos in the current catalog if needed or requested.
        
        Parameters
        ----------
        update: [bool] -optional-
            Shall the computed x and y position be stored in the current data as xpos and ypos columns ?
            
        overwrite: [bool] -optional-
            Shall the update be made if the xpos and ypos columns already exist ?
        
        Returns
        -------
        xy
        """
        xy = np.stack(self.ziff.wcs[0].world_to_pixel_values(np.transpose([self.ra,self.dec]))).T
        if update:
            if 'xpos' not in self.data.keys() or overwrite:
                self.data['xpos'] = xy[0]
                self.data['ypos'] = xy[1]
                
        return xy
    
    def set_mask_pixels(self, mask=None):
        """ set the column corresponding to the entry to be masked out. 0 kept, 1 removed.
        Loaded from ziff if mask is None (self.load_ziffmask())
        """
        if mask is None:
            self.load_ziffmask()
        else:
            self.data["has_badpix"] = mask
                
    def get_xfit(self):
        """ """        
        # return self.xpos.query("filter in [1]")
        return self.xpos[ self.filterflag ]

    def get_yfit(self):
        return self.ypos[ self.filterflag ]

    def get_key_fit(self, key):
        """ """
        if key in self.data.keys():
            return self.data[key].values[ self.filterflag ]
        elif hasattr(self, key):
            return getattr(self,key)[ self.filterflag ]
        raise ValueError('key {} not found'.format(key))

    def get_config(self):
        """ returns the current configuration """
        config = {}
        config['name'] = self.name
        config['filters'] = self._filters
        return config

    # -------- #
    #  LOADER  #
    # -------- #
    def load_ziffmask(self):
        """ """
        rsize = self.ziff.config['i/o']['stamp_size']/2
        self.data['has_badpix'] = 0
       
        for (index, x, y) in zip(self.data.index, self.xpos,self.ypos) :
            mask_ = self.ziff.mask[0].T[(x-rsize).astype(int): (x +rsize).astype(int), (y-rsize).astype(int): (y +rsize).astype(int)]
            if mask_.any() == True:
                self.data.loc[index, 'has_badpix'] = 1
        
    #--------- #
    # MATCHING #
    #--------- #
    def match(self, catalog, seplimit = 1, filtered = True):
        """ """
        skcatalog = catalog.get_skycoord(filtered = filtered)
        sk = self.get_skycoord(filtered = filtered)

        catalog_idx, self_idx, d2d, d3d = search_around_sky(skcatalog, sk, seplimit=seplimit*u.arcsec)
        return self_idx, catalog_idx

    def set_is_isolated(self):
        """ """
        idx1,idx2 = self.match(self, seplimit = 8)
        unique, counts = np.unique(idx1, return_counts=True)
        self.data['is_isolated'] = 0
        wh = unique[counts == 1]
        index = self.data.index[wh]
        self.data.loc[index,'is_isolated'] = 1
        
    #---------- #
    # FILTERING #
    #---------- #
    def update_filter(self):
        """ """
        self.data.loc[:, 'filter'] = 1
        for _filter in self._filters:
            self.data.loc[:, 'filter'] *= self.data.loc[:,_filter]
    
    def add_filter(self, key, range_values, name = None):
        """ """
        if name is None:
            name = key + str(range_values)
        self.data[name] = 0
        if key in self.data.keys():
            values = self.data[key]
        elif hasattr(self, key):
            values = getattr(self, key)
        else:
            raise ValueError("key {} not in keys or attributes ".format(key))
        index = np.logical_and(values >= range_values[0],values < range_values[1])
        self.data.loc[index,name] = 1
        self._filters[name] = {}
        self._filters[name]['range'] = range_values
        self._filters[name]['key'] = key
        self.update_filter()

    def remove_filter(self, name):
        """ """
        if name in self.data.keys():
            self.data.drop(name, axis=1, inplace = True)
            self._filters.pop(name)
            self.update_filter()
        else:
            raise ValueError("Filter {} not found in dataframe.".format(name))
    
    # -------- #
    #   I/O    #
    # -------- #
    def get_datahdu(self, filtered = True):
        cols = []
        df = self.data.reset_index()
        if filtered:
            df = df.loc[self.filterflag]
            
        for _key in df.keys():
            format = 'K' if df[_key].dtype == 'int' else 'D'
            cols.append(fits.Column(name = _key, array = df[_key], format = format, ascii = False))
            
        return fits.BinTableHDU.from_columns(cols)

    def get_primary_hdu(self):
        # Same as ztfcat
        print("DEPRECATED")
        return fits.open(self.ziff.ztfcat[0])[0]

    
    def save_fits(self, savefile, filtered=True, overwrite=True, **kwargs):
        """ """
        print("DEPRECATED, used writeto")
        self.writeto(savefile, filtered=filtered, overwrite=overwrite, **kwargs)

        
    def write_to(self, savefile, filtered=True, format=None, overwrite=True, safeexit=False, header=None):
        """ generic saving function calling the dedicated format ones (to_fits, to_csv)"""
        if os.path.isfile(savefile):
            warnings.warn(f"File {savefile} already exists")
            if not overwrite:
                if safeexit:
                    return None
                raise IOError("Cannot overwrite existing file.")
        
        if format is None:
            format = os.path.splitext(savefile)[-1]
            if format is None:
                raise ValueError("No extension given in savefile and format=None.")
        else:
            if format.startswith("."):
                savefile = savefile+format
            else:
                savefile = savefile+"."+format

        # - 
        if format in ["fits",".fits"]:
            self.to_fits(savefile, filtered=filtered, overwrite=overwrite, header=header)
        else:
            raise ValueError("Only fits format implemented")

    def to_fits(self, savefile, header=None, filtered=True, overwrite=False):
        """ Store the catalog as a fits file. 
        
        Parameters
        ----------
        
        
        Returns
        -------
        exit of fits.HDUList.writeto()
        """
        hdul = []
        # -- Data saving
        if header is None:
            header = fits.Header()
        elif type(header) is dict:
            header = fits.Header(header)
        elif type(header) is not fits.Header:
            raise TypeError(f"input header must be a dict or a fits.Header, {type(header)} given.")
        
        # - Primary
        hdul.append(fits.PrimaryHDU([], header))
        # - Data
        hdul.append(self.get_datahdu(filtered=filtered))
        # -> out
        hdul = fits.HDUList(hdul)
        return hdul.writeto(savefile, overwrite=overwrite)
        
        
    def load_fits(self, path):
        """ """
        print("load_fits DEPRECATED, use the class function")
        f = fits.open(path)
        self.set_data(pd.DataFrame(f[1].data).set_index('Source'))

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def name(self):
        """ Name of the catalog """
        return self._name

    @property
    def data(self):
        """ catalog data """
        return self._data#get_data()
        
    def hasdata(self):
        """ test if the data as been set."""
        return hasattr(self, '_data')

    @property
    def dataframe(self):
        """ """
        print("self.dataframe is DEPRECATED, use self.data")
        return self.data #self._dataframe

    @property
    def df(self):
        """ Shortcut for self.dataframe """
        print("self.data is DEPRECATED, use self.data")
        return self.data # self.dataframe


    @property
    def filterflag(self):
        """ boolean array corresponding to 
        np.asarray(self.data['filter'], dtype="bool")
        """
        return np.asarray(self.data['filter'], dtype="bool")
    @property
    def filtered_index(self):
        """ dataframe index of the data filtered """
        return self.data.loc[self.filterflag].index

    @property
    def filtered_iindex(self):
        return np.where( self.filterflag )[0]
    
    @property
    def xpos(self):
        """ x ccd position, shortcut of self.get_xpos()"""
        return self.get_xpos()

    @property
    def ypos(self):
        """ y ccd position, shortcut of self.get_ypos()"""        
        return self.get_ypos()

    @property
    def ra(self):
        """ ra coordinate, shortcut of self.get_ra()"""
        return self.get_ra()

    @property
    def dec(self):
        """ declination coordinate, shortcut of self.get_dec()"""        
        return self.get_dec()

    @property
    def skycoord(self):
        """ astropy SkyCoord of ra and dec """
        return SkyCoord(self.ra,self.dec, unit = u.deg)
    
    @property
    def ziff(self):
        """ attached ziff object. """
        return self._ziff

######################
#                    #
#  Derived Catalogs  #
#                    #
######################

class ReferenceCatalog(Catalog):
    
    def __init__(self, ziff, name = None, which = 'GAIA'):
        if name is None:
            name =  which
        super().__init__(ziff = ziff, name = name)
        
        if which in ['GAIA','gaia','Gaia']:
            self._which = 'gaia'
        elif which in ['PS','PS1','PanStarrs']:
            raise NotImplementedError('Please contact Melissa.')
        else:
            raise NotImplementedError("Only Gaia supported")

    # ------------ #
    #  Download    #
    # ------------ #
    def download(self, **kwargs):
        """ Dowloads the catalog. """
        # Sometimes there is an issue with the query, which leads to 0 entries and an index error. In this case we just retry once
        retry = kwargs.get("retry", True)
        
        if self._which == 'gaia':
            try:
                df = self.fetch_gaia_catalog(**kwargs).to_pandas().set_index('Source')
            except IndexError:
                if retry:
                    print("Retrying to download gaia cats")
                    self.download(**{**kwargs,**{"retry":False}})
                else:
                    warnings.warn("gaia catalog downloading failed")
                    return
        else:
            raise NotImplementedError(" Only gaia catalog downloading has been implemented ")
        
        self.set_data(df)
                
    
    def get_config(self):
        """ returns the current configuration """
        config = super().get_config()
        config['which'] = self._which
        return config
    
    def fetch_gaia_catalog(self, radius= 0.75, r_unit="deg",column_filters={'Gmag': '10..20'},**kwargs):
        """ query online gaia-catalog in Vizier (I/345, DR2) using astroquery.
        This function requieres an internet connection.
        
        Parameters
        ----------
        center: [string] 'ra dec'
        position of the center of the catalog to query.
        (we use the radec of center of the quadrant)
        
        radius: [string] 'value unit'
        radius of the region to query. For instance '1d' means a
        1 degree raduis
        (from the center of the quadrant to the border it is about 0.65 deg, ask Philippe for details if)

        extracolumns: [list-of-string] -optional-
        Add extra column from the V/139 catalog that will be added to
        the basic query (default: position, ID, object-type, magnitudes)
        column_filters: [dict] -optional-
        Selection criterium for the queried catalog.
        (we have chosen G badn, it coers from 300 to 1000 nm in wavelength)

        **kwargs goes to astroquery.vizier.Vizier

        Returns
        -------
        GAIA Catalog (child of Catalog)
        """
        from astroquery import vizier
        columns = ["Source","RA_ICRS","e_RA_ICRS","DE_ICRS","e_ED_ICRS", "Gmag", "RPmag", "BPmag"]
        #for band in SDSS_INFO["bands"]:

        #try:
        coord = SkyCoord(ra=self.ziff.ra[0],dec=self.ziff.dec[0], unit=(u.deg,u.deg))
        angle = Angle(radius,r_unit)
        v = vizier.Vizier(columns, column_filters=column_filters)
        v.ROW_LIMIT = -1
        # cache is False is necessary, notably when running in a computing center.
        t = v.query_region(coord, radius=angle,catalog="I/345/gaia2", cache=False).values()[0]
        t['colormag'] = t['RPmag'] - t['BPmag']
        return t
    

class BaselineCatalog(ReferenceCatalog):
    def __init__(self, ziff, name = None, which = 'GAIA'):
        super().__init__(ziff, name, which)
        self.download()
        self.set_is_isolated()
        self.add_filter('Gmag',[13,15], name = 'mag_filter')
        self.add_filter('xpos',[100,2900], name = 'border_filter_x')
        self.add_filter('ypos',[100,2900], name = 'border_filter_y')
        self.add_filter('is_isolated',[1,2], name = 'isolated_filter')
    
    
# End of catalog.py ========================================================

#  LocalWords:  toprimary
