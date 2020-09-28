#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as N
import pandas as pd
import astropy.io as aio
import astropy.units as u
import os
from astropy.coordinates import Angle, SkyCoord, search_around_sky
from astropy.wcs import WCS
import copy

class Catalog(object):
    def __init__(self, ziff, name):
        self._name = name
        self._ziff = ziff
        self._filters = {}


    def copy(self, name = None):
        if name is None:
            name = name
        c = Catalog(self._ziff, self._name)
        c._dataframe = self._dataframe.copy()
        c._filters = copy.deepcopy(self._filters)
        c.update_filter()
        return c
        
    def change_name(self, new_name):
        self._name = new_name
    
    def set_astropy_table(self, table):
        raise NotImplementedError("Astropy table not supported")


    def set_dataframe(self, df):
        self._dataframe = df
        for k in df.keys():
            self._dataframe[k] = self._dataframe[k].values.byteswap().newbyteorder()
        self._dataframe['filter'] = 1

    def set_data(self, data):
        if type(data) == pd.DataFrame:
            self.set_dataframe(data)
        else:
            raise NotImplementedError("Type {} not implemented".format(type(data)))

    def __str__(self):
        
        out = "{} object \n".format(self.__class__.__name__)
        out += "Name  : {}\n".format(self._name)
        if hasattr(self, '_dataframe'):
            out += "Number of stars : {}".format(N.size(self.df.loc[self.df['filter']==1],axis=0))
        return out

    def hasdata(self):
        if hasattr(self, '_dataframe'):
            return True
        return False

    def get_data(self, return_type = 'df'):
        """
        return_type = [df, astropy]
        """
        
        if not self.hasdata():
            raise ValueError("You must set_astropy_table(), set_dataframe() or set_data()")
        if return_type == 'df':
            return self.dataframe
        raise ValueError("return_type = {} not udnerstood. Only df possible so far.".format(return_type))

    def get_ra(self):
        keys = ['RA_ICRS','RA','ra']
        for k in keys:
            try:
                return self.df[k]
            except:
                pass
        raise ValueError("Keys {} not in dataframe".format(keys))
    
    def get_dec(self):
        keys = ['DE_ICRS','DE','de','DEC','dec']
        for k in keys:
            try:
                return self.df[k]
            except:
                pass
        raise ValueError("Keys {} not in dataframe".format(keys))

    def get_xpos(self):
        # If in keys, used keys
        keys = ['x','xpos']
        for k in keys:
            try:
                return self.df[k].values
            except:
                pass
        # Else compute it
        return self.get_xy_from_radec()[0]

    def set_sky(self):
        bkgd = self.ziff.get_ztfimg()[0].get_background()
        self.df['sky'] = bkgd[N.clip(self.xpos,0,3000).astype(int),N.clip(self.ypos,0,3000).astype(int)]
        
    def get_ypos(self):
        # If in keys, used keys
        keys = ['y','ypos']
        for k in keys:
            try:
                return self.df[k].values
            except:
                pass
        # Else compute it
        # Note that we could save them but for now we don't
        return self.get_xy_from_radec()[1]

    def get_xy_from_radec(self):
        xy = N.stack(self.ziff.wcs[0].world_to_pixel_values(N.transpose([self.ra,self.dec]))).T
        if 'xpos' not in self.df.keys():
            self.df['xpos'] = xy[0]
            self.df['ypos'] = xy[1]
        return xy

    
    def set_mask_pixels(self):
        """ """
        rsize = self.ziff.config['i/o']['stamp_size']/2
        self.df['has_badpix'] = 0
       
        for (index, x, y) in zip(self.df.index, self.xpos,self.ypos) :
            mask_ = self.ziff.mask[0].T[(x-rsize).astype(int): (x +rsize).astype(int), (y-rsize).astype(int): (y +rsize).astype(int)]
            if mask_.any() == True:
                self.df.loc[index, 'has_badpix'] = 1
                
    def get_xfit(self):
        return self.xpos[self.df['filter'].values==1]

    def get_key_fit(self, key):
        if key in self.df.keys():
            return self.df[key].values[self.df['filter'].values==1]
        elif hasattr(self, key):
            return getattr(self,key)[self.df['filter'].values==1]
        raise ValueError('key {} not found'.format(key))

    def get_yfit(self):
        return self.ypos[self.df['filter'].values==1]

    def get_skycoord(self, filtered = False):
        if filtered:
            return SkyCoord(self.ra.values[self.filtered_iindex],self.dec.values[self.filtered_iindex], unit = u.deg)
        return SkyCoord(self.ra.values,self.dec.values, unit = u.deg)
        

    ############
    # MATCHING #
    ############
    def match(self, catalog, seplimit = 1, filtered = True):
        skcatalog = catalog.get_skycoord(filtered = filtered)
        sk = self.get_skycoord(filtered = filtered)

        catalog_idx, self_idx, d2d, d3d = search_around_sky(skcatalog, sk, seplimit=seplimit*u.arcsec)
        return self_idx, catalog_idx

    def set_is_isolated(self):
        idx1,idx2 = self.match(self, seplimit = 8)
        unique, counts = N.unique(idx1, return_counts=True)
        self.df['is_isolated'] = 0
        wh = unique[counts == 1]
        index = self.df.index[wh]
        self.df.loc[index,'is_isolated'] = 1
        
    #############
    # FILTERING #
    #############

    def add_filter(self, key, range_values, name = None):
        if name is None:
            name = key + str(range_values)
        self.df[name] = 0
        if key in self.df.keys():
            values = self.df[key]
        elif hasattr(self, key):
            values = getattr(self, key)
        else:
            raise ValueError("key {} not in keys or attributes ".format(key))
        index = N.logical_and(values >= range_values[0],values < range_values[1])
        self.df.loc[index,name] = 1
        self._filters[name] = {}
        self._filters[name]['range'] = range_values
        self._filters[name]['key'] = key
        self.update_filter()

    def remove_filter(self, name):
        if name in self.df.keys():
            self.df.drop(name, axis=1, inplace = True)
            self._filters.pop(name)
            self.update_filter()
        else:
            raise ValueError("Filter {} not found in dataframe.".format(name))
        
    def update_filter(self):
        self.df.loc[:, 'filter'] = 1
        for _filter in self._filters:
            self.df.loc[:, 'filter'] *= self.df.loc[:,_filter]

    #######
    # I/O #
    #######

    def get_hdu(self, filtered = True):
        cols = []
        df = self.df.reset_index()
        if filtered:
            df = df.loc[df['filter'] == 1]
        for _key in df.keys():
            if df[_key].dtype == 'int':
                format = 'K'
            else:
                format = 'D'
            cols.append(aio.fits.Column(name = _key, array = df[_key], format = format, ascii = False))
        return aio.fits.BinTableHDU.from_columns(cols)


    def get_primary_hdu(self):
        # Same as ztfcat
        return aio.fits.open(self.ziff.ztfcat[0])[0]
    
    def save_fits(self,  suffix = None, path = None,filtered = True, overwrite = True):
        if path is None:
            path = self.ziff.prefix[0]+suffix
        if os.path.isfile(path):
            print("WARNING: File {} already exists".format(path))
            if overwrite:
                print("WARNING: Overwritting {}".format(path))
                os.remove(path)
        hdul = aio.fits.HDUList([self.get_primary_hdu(),self.get_hdu(filtered=filtered)])
        hdul.writeto(path)
        return

    
    def load_fits(self, path):
        f = aio.fits.open(path)
        self.set_dataframe(pd.DataFrame(f[1].data).set_index('Source'))

       

    def get_config(self):
        config = {}
        config['name'] = self.name
        config['filters'] = self._filters
        return config
    
    @property
    def data(self):
        return self.get_data()
    
    @property
    def dataframe(self):
        return self._dataframe

    @property
    def df(self):
        """ Shortcut for self.dataframe """
        return self.dataframe

    @property
    def filtered_index(self):
        return self.df.loc[self.df['filter']==1].index

    @property
    def filtered_iindex(self):
        return N.where(self.df['filter']==1)[0]
    
    @property
    def xpos(self):
        return self.get_xpos()

    @property
    def ypos(self):
        return self.get_ypos()

    
    @property
    def ra(self):
        return self.get_ra()

    @property
    def dec(self):
        return self.get_dec()

    @property
    def skycoord(self):
        return SkyCoord(self.ra,self.dec, unit = u.deg)
    
    @property
    def ziff(self):
        return self._ziff

    @property
    def name(self):
        return self._name

    
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
        

    def download(self, **kwargs):
        # Sometimes there is an issue with the query, which leads to 0 entries and an index error. In thiscase we just retry.
        if self._which == 'gaia':
            fail = True
            while fail==True:
                try:
                    df = self.fetch_gaia_catalog(**kwargs).to_pandas().set_index('Source')
                    fail = False
                except IndexError:
                    print("Retrying to download gaia cats")
                    pass
            self.set_dataframe(df)

    
    def get_config(self):
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
        

class ZTFCatalog(Catalog):
    def __init__(self, ziff, name = 'ztfcat'):
        super().__init__(ziff = ziff, name = name)
        self.load_ZTF_catalog()
        
    def load_ZTF_catalog(self):
        f = aio.fits.open(self.ziff.ztfcat[0])
        self.set_dataframe(pd.DataFrame(f[1].data).set_index('sourceid'))
        
    
    
# End of catalog.py ========================================================
