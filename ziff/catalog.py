#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import warnings

import pandas
import numpy as np

from astropy import units
from astropy.io import fits #as aio
from astropy.coordinates import Angle, SkyCoord, search_around_sky

# out for Dask
from .utils import avoid_duplicate
from ztfimg.stamps import stamp_it

from ztfquery.io import CCIN2P3
_CC = CCIN2P3(connect=False)


def fetch_ziff_catalog(ziff, which="gaia", as_collection=True, **kwargs):
    """ High level function that fetch the `which` catalog data for the given ziff.
    returns a CatalogCollection if allowed if the ziff is not a single ziff. 
    
    **kwargs goes to fetch_{which}_catalog

    Returns
    -------
    Catalog or CatalogCollection
    """
    if ziff.is_single():
        ra,dec = ziff.get_center(inpixel=False)
        radius = ziff.get_diagonal(inpixel=False)/1.7 # Not 2 to have some wiggle room
        return fetch_catalog(which, ra, dec, radius, r_unit="deg", **kwargs)
    else:
        coords = ziff.get_center(inpixel=False)
        radii  = np.asarray(ziff.get_diagonal(inpixel=False))/1.7
        cats = []
        for (ra_,dec_), radius_ in zip(coords, radii):
            cats.append( fetch_catalog(which, ra_, dec_, radius_, r_unit="deg", **kwargs) )

    if as_collection:
         return CatalogCollection(cats, load_data=True)
     
    return cats

def fetch_catalog(which, ra, dec, radius, r_unit="deg", **kwargs):
    """ """
    return eval(f"fetch_{which}_catalog")(ra, dec, radius, r_unit=r_unit, **kwargs)

def fetch_gaia_catalog(ra, dec, radius= 0.75, r_unit="deg",
                        column_filters={'Gmag': '10..20'}, name="gaia",
                        queryhost="vizier",
                        catname="I/350/gaiaedr3",
                        **kwargs):
    """ query online gaia-catalog in Vizier (I/350/gaiaedr3, eDR3) using astroquery.
    This function requieres an internet connection.
        
    Parameters
    ----------
    ra, dec: [float]
        center of the Catalog [in degree]

    center: [string] 'ra dec'
    position of the center of the catalog to query.
    (we use the radec of center of the quadrant)
        
    radius: [string] 'value unit'
    radius of the region to query. For instance '1d' means a
    1 degree raduis
    (from the center of the quadrant to the border it is about 0.65 deg)

    extracolumns: [list-of-string] -optional-
    Add extra column from the V/139 catalog that will be added to
    the basic query (default: position, ID, object-type, magnitudes)
    column_filters: [dict] -optional-
    Selection criterium for the queried catalog.
    (we have chosen G badn, it coers from 300 to 1000 nm in wavelength)

    **kwargs goes to Catalog.__init__

    Returns
    -------
    GAIA Catalog (child of Catalog)
    """
    if queryhost == "vizier":
        df = _fetch_gaia_catalog_vizier_(ra, dec, radius= 0.75, r_unit=r_unit,
                                             column_filters=column_filters,
                                             catname=catname)
    elif queryhost == "ccin2p3":
        df = _CC.query_catalog(ra, dec, radius, catname=catname, depth=7, **kwargs)
    
    return Catalog(dataframe=df, name=name, **kwargs)


def _fetch_gaia_catalog_vizier_(ra, dec, radius= 0.75, r_unit="deg",
                                    column_filters={'Gmag': '10..20'},
                                    catname="I/350/gaiaedr3"):
    """ query online gaia-catalog in Vizier (I/350/gaiaedr3, eDR3) using astroquery.
    This function requieres an internet connection.
        
    Parameters
    ----------
    ra, dec: [float]
        center of the Catalog [in degree]

    center: [string] 'ra dec'
    position of the center of the catalog to query.
    (we use the radec of center of the quadrant)
        
    radius: [string] 'value unit'
    radius of the region to query. For instance '1d' means a
    1 degree raduis
    (from the center of the quadrant to the border it is about 0.65 deg)

    extracolumns: [list-of-string] -optional-
    Add extra column from the V/139 catalog that will be added to
    the basic query (default: position, ID, object-type, magnitudes)
    column_filters: [dict] -optional-
    Selection criterium for the queried catalog.
    (we have chosen G badn, it coers from 300 to 1000 nm in wavelength)

    **kwargs goes to Catalog.__init__

    Returns
    -------
    GAIA Catalog (child of Catalog)
    """
    
    from astroquery import vizier
    columns = ["Source","RA_ICRS","e_RA_ICRS","DE_ICRS","e_ED_ICRS", "Gmag", "RPmag", "BPmag",
                   "FG", "FRP", "FBP"]
    
    coord = SkyCoord(ra=ra,dec=dec, unit=(units.deg,units.deg))
    angle = Angle(radius, r_unit)
    v = vizier.Vizier(columns, column_filters=column_filters)
    v.ROW_LIMIT = -1
    # cache is False is necessary, notably when running in a computing center.
    gaiatable = v.query_region(coord, radius=angle, catalog=catname, cache=False).values()[0]
    gaiatable['colormag'] = gaiatable['BPmag'] - gaiatable['RPmag']
    
    return gaiatable.to_pandas().set_index('Source')

def dataframe_to_hdu(dataframe, drop_notimplemented=True):
    """ converts a dataframe into a fits.BinTableHDU """
    # L: Logical (Boolean)
    # B: Unsigned Byte
    # I: 16-bit Integer
    # J: 32-bit Integer
    # K: 64-bit Integer
    # E: Single-precision Floating Point
    # D: Double-precision Floating Point
    # C: Single-precision Complex
    # M: Double-precision Complex
    # A: Character
    
    cols = []
    df = dataframe.reset_index()
    for _key in df.keys():
        type_ = df[_key].dtype
        
        if type_ == "object":
            if drop_notimplemented:
                warnings.warn(f"column type {type_} conversion to fits format not implemented | {_key} droped")
                continue
            raise NotImplementedError(f"column type {type_} conversion to fits format not implemented")
        
        if type_ in ['int','int64',"Int64"]:
            format = 'K'
            value = df[_key].astype('int')
        elif type_ == 'float':
            format = 'D'
            value = df[_key].astype('float')
        elif type_ == 'bool' or type_ == 'boolean':
            format = 'L'
            value = df[_key].astype('bool')
        elif type_ == 'string':
            format = 'A'
            value = df[_key].astype('string')
        else:
            if drop_notimplemented:
                warnings.warn(f"column type {type_} conversion to fits format not implemented | {_key} droped")
                continue
            raise NotImplementedError(f"column type {type_} conversion to fits format not implemented")
        
        cols.append(fits.Column(name=_key, array=value, format=format, ascii=False))
            
    return fits.BinTableHDU.from_columns(cols)

    
######################
#                    #
#     Catalog        #
#                    #
######################
class Catalog(object):
    
    def __init__(self, dataframe=None, name=None, wcs=None, header=None, mask=None,
                     xyformat=None):
        """ """
        self._name = name
        self._filters = {}
        
        if dataframe is not None:
            self.set_data( dataframe )

        if wcs is not None:
            self.set_wcs(wcs)

        if header is not None:
            self.set_header(header)

        if mask is not None:
            self.set_mask(mask)

        self._xyformat = xyformat
            
    def __str__(self):
        """ printing method """
        out = "{} object \n".format(self.__class__.__name__)
        out += "Name  : {}\n".format(self._name)
        if hasattr(self, '_data'):
            out += "Number of stars : {}".format(np.size(self.data.loc[~self.filterout],axis=0))
        return out

    def copy(self, name = None, **kwargs):
        """ """        
        return self.get_catalog(name=name, **kwargs)
    
    # ----- #
    #  I/O  #
    # ----- #
    @classmethod
    def load(cls, filename, name=None, wcs=None, header=None, mask=None, multi_ok=True, **kwargs):
        """ Generic loading method.
        

        Parameters
        ----------
        
        """
        filename = np.atleast_1d(filename)
        if len(filename)>1:
            if not multi_ok:
                raise ValueError("Only single filename could be used to set a catalog. use MultiCatalog.load or set multi_ok=True for having this automated")
            return MultiCatalog.load(filename, name=name, wcs=wcs,
                                     header=header, mask=mask, **kwargs)

        filename = filename[0]
        extension = filename.split(".")[-1]
        if extension in ["csv"]:
            return cls.read_cvs(read_cvs, name=name, wcs=wcs, header=header, mask=mask, **kwargs)
        
        if extension in ["fits"]:
            return cls.read_fits(read_cvs, name=name, wcs=wcs, header=header, mask=mask, **kwargs)

        raise ValueError("only csv and fits loading implemented.")
            
    @classmethod
    def read_cvs(cls, filename, name="catalog", index_col=None, readprop={}, **kwargs):
        """ """
        return cls( dataframe=pandas.read_csv(filename, index_col=index_col, **readprop),
                        name=name, **kwargs)

    @classmethod
    def read_fits(cls, filename, dataext=1, headerext=None, name="catalog",
                      index_col='Source', **kwargs):
        """ """
        from astropy import table
        data = fits.getdata(filename, ext=dataext)
        if headerext is None:
            headerext = dataext
        header = fits.getheader(filename, ext=headerext)
        dataframe = table.Table(data).to_pandas()
        if index_col is not None:
            dataframe = dataframe.set_index(index_col)
            
        this = cls(dataframe, name=name, **kwargs)
        this.set_header(header)
        return this
    
    @classmethod
    def read_psfcat(cls, psfcat, name="ztfcat"):
        """ loads from ziff.ztfcat[0] """
        return cls.read_fits(psfcat, name="ztfcat", index_col='sourceid')


    def write_to(self, savefile, filtered=True, extension=None, overwrite=True,
                     safeexit=False, store_filename=True, **kwargs):
        """ generic saving function calling the dedicated format ones (to_fits, to_csv) """
        
        if os.path.isfile(savefile):
            
            if not overwrite:
                if safeexit:
                    warnings.warn(f"File {savefile} already exists ; overwrite is False")
                    return None
                raise IOError("Cannot overwrite existing file.")
        
        if extension is None:
            extension = os.path.splitext(savefile)[-1]
            if extension is None:
                raise ValueError("No extension given in savefile and extension=None.")
        else:
            if extension.startswith("."):
                savefile = savefile+extension
            else:
                savefile = savefile+f".{extension}"

        # - 
        if extension in ["fits",".fits"]:
            self.to_fits(savefile, filtered=filtered, overwrite=overwrite, store_filename=store_filename,**kwargs)
        elif extension in ["csv",".csv"]:
            self.to_csv(savefile, filtered=filtered, overwrite=overwrite, store_filename=store_filename, **kwargs)
        else:
            raise ValueError("Only fits and csv format implemented")

    def to_fits(self, savefile, header=None, filtered=True, overwrite=False, shuffled=False, store_filename=True):
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
        hdul.append( self.get_data(filtered=filtered, as_hdu=True, shuffled=shuffled) )
        # -> out
        hdul = fits.HDUList(hdul)
        
        out =  hdul.writeto(savefile, overwrite=overwrite)
        if store_filename:
            self.set_filename(savefile)
            
        return out

    def to_csv(self, savefile, filtered=True, overwrite=False, shuffled=False, store_filename=True, **kwargs):
        """ """
        if os.path.isfile(savefile) and not overwrite:
            raise IOError(f"Cannot overwrite {savefile}")
        df = self.get_data(filtered=filtered, shuffled=shuffled).reset_index()
        out = df.to_csv(savefile, **kwargs)
        if store_filename:
            self.set_filename(savefile)
            
        return out
        
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
        if type(dataframe) != pandas.DataFrame:
            try:
                dataframe = pandas.DataFrame(dataframe)
            except:
                raise TypeError("The input dataframe is not a DataFrame and cannot be converted into one.")

        # reshaped:
        self._data = pandas.DataFrame(dataframe.values.byteswap().newbyteorder(),
                                      index=dataframe.index, columns=dataframe.columns
                                     ).convert_dtypes() # fixes object dtype issues
                                     
        if "Int64" in np.asarray(self._data.dtypes, dtype="str"):
            self._data.astype(self._data.dtypes.replace('Int64','int64')) # avoids warnings
            
        if 'filterout' not in self._data.columns:
            self._data['filterout'] = False
    
    def set_wcs(self, wcs):
        """ Attach an astropy WCS solution to the catalog. """
        self._wcs = wcs

    def set_header(self, header):
        """ """
        if header is None:
            self._header = fits.Header()
        elif type(header) is dict:
            self._header = fits.Header(header)
        elif type(header) is not fits.Header:
            raise TypeError(f"input header must be a dict or a fits.Header, {type(header)} given.")

    def set_mask(self, mask):
        """ set the column corresponding to the entry to be masked out. 0 kept, 1 removed. """
        self._mask = mask
        if self.has_data():
            self.data["masked"] = mask

    def set_skybackground(self, bkgd, askey="sky"):
        """ """
        setattr(self,f"_{askey}", bkgd)
        if self.has_data():
            self.data[askey] = bkgd

    def set_filename(self, filename, ignore_warning=False):
        """ """
        if self.has_filename() and not ignore_warning:
            warnings.warn("you are overwritting the current filename {self.filename} by {filename}")
            
        self._filename = filename
        
    # -------- #
    #  LOADER  #
    # -------- #
    def build_filename(self, prefix="", extension=".fits"):
        """ returns prefix+self.name+extension """
        if not extension.startswith("."):
            extension = f".{extension}"
            
        return prefix+self.name+extension
        
    def build_sky_from_bkgdimg(self, bkgdimg, stampsize, askey="sky",
                                   npfunc="nanmean"):
        """ build background entry based on bkgdimg given the stamp size.
        
        Parameters
        ----------
        bkgdimg: [2d-array]
            Background image.

        stampsize: [int]
            Size of the stamps. 
            Presence of True in maskimg will be looked for for any catalog entry 
            [{x/y}-stampsize/2, {x/y}+stampsize/2]

        askey: [string] -optional-
            The column name inside the dataframe

        npfunc: [string] -optional-
            Which numpy function should be used to go from a background stamp into a unique background ?
                    
        Returns
        -------
        None (set_skybackground())

        Example
        -------
        if you have a ziff:
        self.build_sky_from_bkgdimg(ziff.get_background(), ziff.get_config_value("stamp_size", squeeze=True))
        """
        skystamp = self.get_datastamps(bkgdimg, stampsize=stampsize, filtered=False)
        sky = getattr(np,npfunc)(skystamp, axis=(1,2))
        # - Setting the sky
        self.set_skybackground(sky, askey=askey)
    
    def build_mask_from_maskimg(self, maskimg, stampsize):
        """ build catalog mask based on maskimg given the stamp size.
        
        Parameters
        ----------
        maskimg: [2d-array boolean]
            Mask image where True means to be masked out

        stampsize: [int]
            Size of the stamps. 
            Presence of True in maskimg will be looked for for any catalog entry 
            [{x/y}-stampsize/2, {x/y}+stampsize/2]

        Returns
        -------
        None (set_mask)

        Example
        -------
        if you have a ziff:
        self.build_mask_from_maskimg(ziff.mask, ziff.get_config_value("stamp_size", squeeze=True))
        """
        maskstamp = np.asarray(self.get_datastamps(maskimg, stampsize=stampsize, filtered=False), dtype="bool")
        maskout   = np.any(maskstamp, axis=(1,2))
        # Setting the masks
        self.set_mask( maskout )

    def load_xy_from_radec(self, update=True, overwrite=False, returns=False, wcs=None):
        """ computes the x and y position given the wcs solution and the radec values. 
        Store them as xpos and ypos in the current catalog if needed or requested.
        
        Parameters
        ----------
        update: [bool] -optional-
            Shall the computed x and y position be stored in the current 
            data as xpos and ypos columns ?
            
        overwrite: [bool] -optional-
            Shall the update be made if the xpos and ypos columns already exist ?
        
        wcs: [astropy.wcs] -optional-
            WCS used for the conversion. If None given self.wcs will be used if it exists.
            = remark, if given, this *does not* call set_wcs() =

        Returns
        -------
        xy
        """
        if wcs is None:
            if not self.has_wcs():
                raise AttributeError("no wcs set, no wcs given")
            wcs = self.wcs
        
        x,y = np.asarray( wcs.world_to_pixel_values(self.get_ra(filtered=False),
                                                    self.get_dec(filtered=False))
                        )
        if update:
            if 'xpos' not in self.data.keys() or overwrite:
                self.data['xpos'] = x
                self.data['ypos'] = y
        if returns:
            return x,y

    def load_radec_from_xy(self, update=True, overwrite=False, returns=False, wcs=None):
        """ computes the ra and dec position given the wcs solution and the xpos ypos values. 
        Store them as ra and dec in the current catalog if needed or requested.
        
        Parameters
        ----------
        update: [bool] -optional-
            Shall the computed ra and dec position be stored in the current 
            data as 'ra' and 'dec' columns ?
            
        overwrite: [bool] -optional-
            Shall the update be made if the xpos and ypos columns already exist ?
        
        wcs: [astropy.wcs] -optional-
            WCS used for the conversion. If None given self.wcs will be used if it exists.
            = remark, if given, this *does not* call set_wcs() =

        Returns
        -------
        ra,dec
        """
        if wcs is None:
            if not self.has_wcs():
                raise AttributeError("no wcs set, no wcs given")
            wcs = self.wcs
            
        ra, dec = np.asarray( wcs.pixel_to_world_values(self.get_xpos(filtered=False),
                                                        self.get_ypos(filtered=False))
                            )
        if update:
            if 'ra' not in self.data.keys() or overwrite:
                self.data['ra'] = ra
                self.data['dec'] = dec
                
        if returns:
            return ra,dec

    def _guess_xyformat_(self):
        """ """
        # not yet RA/Dec or XPOS/YPOS, setting numpy as default
        if not self._xposkey in self.data or not self._rakey in self.data:
            self._xyformat = "numpy"
        # aldeady one, let's see which
        else:
            np_xpos, np_ypos = self.wcs.world_to_pixel_values(self.get_ra(filtered=False),
                                                       self.get_dec(filtered=False))
            shift = np.unique(self.data[self._xposkey] - np_xpos)
            
            if len(shift)>1:
                raise ValueError(f"non-constant offset when guessing the xyformat. This unexpected. shift: {shift}")
            if len(shift)==0:
                return None
            if int(shift[0])==0:
                self._xyformat = "numpy"
            elif int(shift[0])==1: # to be checked.
                self._xyformat = "fortran"
            else:
                raise ValueError("non 0 or 1 origin when guessung the xyformat. This unexpected.")
            
    def _get_xyorigin_(self, xyformat):
        """ """
        if xyformat not in ["numpy", "matplotlib", "fortran", "fits"]:
            raise ValueError("xyformat can only be (numpy, matplotlib ; origin=0) or (fortran, fits; origin=1 )")

        if self.xyformat == "numpy" and xyformat in ["numpy", "matplotlib"] or \
           self.xyformat == "fortran" and xyformat in ["fortran", "fits"]:
            origin=0
        elif self.xyformat == "numpy" and xyformat in ["fortran", "fits"]:
            origin=-1
        elif self.xyformat == "fortran" and xyformat in ["numpy", "matplotlib"]:
            origin=1
            
        return origin
    # -------- #
    #  GETTER  #
    # -------- #
    
    # - Returns Copy
    def get_catalog(self, filtered=False, shuffled=False, xyformat=None, name=None, index=None, **kwargs):
        """ Get the filtered version of the catalog 
        
        Returns
        ------
        self.__class__
        """
        if name is None:
            name = self.name

        new_data = self.get_data(filtered=filtered, shuffled=shuffled,
                                     xyformat=xyformat, index=index,
                                     as_hdu=False).copy()
        mask = new_data["masked"].values
        if xyformat is None:
            xyformat = self.xyformat
            
        new_cat = self.__class__(new_data,
                                  name=name,
                                  wcs=self.wcs, header=self.header,
                                  mask=mask,
                                  xyformat=xyformat,
                                  **kwargs)
        
        if not filtered:
            new_cat._filters = self._filters.copy()
        else:
            _ = [new_cat.data.pop(k) for k in self._filters.keys()]
            _ = new_cat.data.pop("filterout")
            
        return new_cat
            
    def get_filtered(self, shuffled=False, **kwargs):
        """ Get the filtered version of the catalog 
        
        Returns
        ------
        self.__class__
        """
        return self.get_catalog(filtered=True, shuffled=shuffled, name=f"filtered_{self.name}", **kwargs)

    def get_as_xyformat(self, xyformat, filtered=False, shuffled=False, **kwargs):
        """ """
        return self.get_catalog(xyformat=xyformat, filtered=filtered, shuffled=shuffled,
                                name=f"{xyformat}format_"+ ("filtered_" if filtered else "") +self.name,
                                **kwargs)
        
    # - Return DataFrame
    def get_data(self, filtered=False, shuffled=False, xyformat=None, as_hdu=False, index=None):
        """ Basis of the catalog class. 

        Returns
        -------
        DataFrame
        """
        if not self.has_data():
            raise AttributeError("No data set yet. Use self.set_data()")

        d_ = self.data.copy()
        if len(d_)==0:
            return d_
            
        if xyformat is not None:
            origin = self._get_xyorigin_(xyformat)
            if origin != 0:
                d_[self._xposkey] -= origin
                d_[self._yposkey] -= origin
                
        if filtered:        
            d_ = d_[~self.filterout]

        if index is not None:
            d_ = d_.loc[[i for i in index if i in d_.index]]

        if shuffled:
            d_ = d_.sample(frac=1)
        
        if as_hdu:
            return dataframe_to_hdu(d_)
        
        return d_

    def get_header(self):
        """ fits header """
        if self.header is None:
            return fits.Header()
        return self.header
    
    def get_skycoord(self, filtered=True, **kwargs):
        """ Get RA, Dec as astropy.coordinates.SkyCoord """
        return SkyCoord(self.get_ra(filtered=filtered, asserie=False, **kwargs),
                        self.get_dec(filtered=filtered, asserie=False, **kwargs),
                        unit = units.deg)

    def get_ra(self, compute=True, filtered=True, asserie=True,**kwargs):
        """ Get the Right ascension column """
        if self._rakey is None:
            if compute:
                self.load_radec_from_xy(update=True, returns=False)
            else:
                return None

        serie_ = self.get_data(filtered=filtered, **kwargs)[self._rakey]        
        return serie_ if asserie else serie_.values

    
    def get_dec(self, compute=True, filtered=True, asserie=True,**kwargs):
        """ Get the Declination column """
        if self._deckey is None:
            if compute:
                self.load_radec_from_xy(update=True, returns=False)
            else:
                return None

        serie_ = self.get_data(filtered=filtered, **kwargs)[self._deckey]
        return serie_ if asserie else serie_.values
    
    def get_xpos(self, filtered=True, compute=True, asserie=True, xyformat="numpy", **kwargs):
        """ Get the ccd x position column. This corresponds to images.data[x,y] not data.T[x,y]"""
        # If in keys, used keys
        if self._xposkey is None:
            if compute:
                self.load_xy_from_radec(update=True, returns=False)
            else:
                return None
            
        origin = self._get_xyorigin_(xyformat)
        serie_ = self.get_data(filtered=filtered, **kwargs)[self._xposkey]-origin
        return serie_ if asserie else serie_.values

    def get_ypos(self, filtered=True, compute=True, asserie=True, xyformat="numpy", **kwargs):
        """ Get the ccd y position column. This corresponds to images.data[x,y] not data.T[x,y]"""
        if self._yposkey is None:
            if compute:
                self.load_xy_from_radec(update=True, returns=False)
            else:
                return None
            
        origin = self._get_xyorigin_(xyformat)
        serie_ = self.get_data(filtered=filtered, **kwargs)[self._yposkey]-origin
        return serie_ if asserie else serie_.values

    def get_datastamps(self, array, stampsize, filtered=False, xyformat="numpy"):
        """ """
        xpos = self.get_xpos(filtered=filtered, xyformat=xyformat)
        ypos = self.get_ypos(filtered=filtered, xyformat=xyformat)
        if len(xpos)==0:
            ValueError("size of xpos is zero.")
        return stamp_it(array, xpos, ypos,
                        dx=stampsize, asarray=True)
    
        
    def get_config(self):
        """ returns the current configuration """
        config = {}
        config['name'] = self.name
        config['filters'] = self._filters
        return config

    # -------- #
    #  LOADER  #
    # -------- #
    #--------- #
    # MATCHING #
    #--------- #
    def match(self, catalog, seplimit = 1*units.arcsec, filtered = False):
        """ """
        skcatalog = catalog.get_skycoord(filtered = filtered)
        sk = self.get_skycoord(filtered = filtered)

        catalog_idx, self_idx, d2d, d3d = sk.search_around_sky(skcatalog,
                                                            seplimit=seplimit)
        return self_idx, catalog_idx

    def measure_isolation(self, refcat=None, seplimit=8):
        """ 
        
        Parameters
        ----------
        refcat: [Catalog or None] -optional-
            Refence catalog. Isolation will be measured as matching of self with this catalog.
            = IMPORTANT, refcat must include at least *all* the current catalog entries =
            If None, self will be used (self isolation).

        seplimit: [float] -optional-
            isolation distance in arcsec
        
        Returns
        -------
        
        """
        if refcat is None:
            refcat = self
            
        idx1, idx2 = self.match(refcat, seplimit=seplimit*units.arcsec, filtered=False)
        unique, counts = np.unique(idx1, return_counts=True)
        self.data["n_nearsources"] = counts-1
        self.data['is_isolated'] = (self.data["n_nearsources"]==0)        
    #---------- #
    # FILTERING #
    #---------- #
    def update_filter(self, reset=True, used_filters=None):
        """ """
        if used_filters is None:
            used_filters = list(self._filters.keys())
        if 'filterout' in self.data.columns and not reset:
            used_filters += ["filterout"]
            
        self.data.loc[:, 'filterout'] = self.data[used_filters].sum(axis=1).astype("bool")
    
    def add_filter(self, key, range_values, name = None, update=True, verbose=False):
        """ """
        if name is None:
            name = key + str(range_values)

        if key in ["xpos","ypos"] and key not in self.data.keys():
            self.load_xy_from_radec(update=True, returns=False)
        elif key in ["ra","dec"] and key not in self.data.keys():
            self.load_radec_from_xy(update=True, returns=False)
            
        self.data[name] = False

        if len(np.atleast_1d(range_values))==2:
            if verbose: print(f"{key} between {range_values}")
            self.data.loc[:,name] = ~self.data[key].between(*range_values)
        elif len(np.atleast_1d(range_values))==1:
            if verbose: print(f"{key} equals {range_values}")
            self.data.loc[:,name] = ~(self.data[key] == np.atleast_1d(range_values)[0])
        else:
            raise ValueError("cannot parse the given range_values, should have size 1 or 2")
        
        self._filters[name] = {'range':range_values,
                               'key':key}
        if update:
            self.update_filter()

    def remove_filter(self, name):
        """ """
        if name in self.data.keys():
            self.data.drop(name, axis=1, inplace = True)
            self._filters.pop(name)
            self.update_filter()
        else:
            raise ValueError(f"Filter {name} not found in dataframe.")
    
    # --------- #
    #  Internal #
    # --------- #
    def _fetch_datakey_(self, trialkeys, safeout=False):
        """ """
        if not self.has_data():
            raise AttributeError("No data set.")
        
        trialkeys = np.asarray(trialkeys)
        keyin = np.isin(trialkeys, self.data.columns)
        if not np.any(keyin):
            if safeout:
                return None
            raise ValueError(f"None of {trialkeys} are data columns")
        if len(keyin[keyin])>1:
            warnings.warn(f"Several of {trialkeys} are in data columns, first used.")
            
        return trialkeys[keyin][0]
        

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def name(self):
        """ Name of the catalog """
        return self._name

    @property
    def filename(self):
        """ """
        if not self.has_filename():
            return None
        return self._filename

    def has_filename(self):
        """ """
        return hasattr(self,"_filename") and self._filename is not None
    
    @property
    def data(self):
        """ catalog data """
        return self._data

    @property
    def npoints(self):
        """ number of data entries """
        return len(self.data) if self.has_data() else None
    
    def has_data(self):
        """ test if the data as been set."""
        return hasattr(self, '_data')

    @property
    def wcs(self):
        """ """
        if not self.has_wcs():
            return None
        return self._wcs

    def has_wcs(self):
        """ """
        return hasattr(self,"_wcs") and self._wcs is not None

    @property
    def header(self):
        """ """
        if not hasattr(self,"_header"):
            return None
        return self._header

    @property    
    def mask(self):
        """ """
        if not hasattr(self,"_mask"):
            return None
        return self._mask
    
    @property
    def filterout(self):
        """ boolean array corresponding to 
        np.asarray(self.data['filterout'], dtype="bool")
        """
        if 'filterout' in self.data:
            return np.asarray(self.data['filterout'], dtype="bool")
        
        return np.asarray(np.zeros(len(self.data)), dtype="bool")
    
    @property
    def filtered_index(self):
        """ dataframe index of the data filtered """
        return self.data.loc[self.filterout].index

    @property
    def filtered_iindex(self):
        return np.where( self.filterout )[0]

    @property
    def xyformat(self):
        """ format for the x,y origin (0: numpy/matplotlib ; 1: fortran/FITS) """
        if not hasattr(self,"_xyformat") or self._xyformat is None:
            self._guess_xyformat_()
            
        return self._xyformat
        
    @property
    def xpos(self):
        """ x ccd position, shortcut of self.get_xpos()"""
        print("xpos DEPRECATED, use self.get_xpos()")
        return self.get_xpos()

    @property
    def _xposkey(self):
        """ The ccd y position key """
        if not hasattr(self,"_hxposkey") or self._hxposkey is None:
            self._hxposkey = self._fetch_datakey_(['X','x','XPOS','xpos'], safeout=True)
        return self._hxposkey

    @property
    def ypos(self):
        """ y ccd position, shortcut of self.get_ypos()"""
        print("ypos DEPRECATED, use self.get_ypos()")
        return self.get_ypos()

    @property
    def _yposkey(self):
        """ The ccd y position key """
        if not hasattr(self,"_hyposkey") or self._hyposkey is None:
            self._hyposkey = self._fetch_datakey_(['Y','y','YPOS','ypos'], safeout=True)
        return self._hyposkey
    
    @property
    def ra(self):
        """ ra coordinate, shortcut of self.get_ra()"""
        return self.get_ra()
    
    @property
    def _rakey(self):
        """ The RA key """
        if not hasattr(self,"_hrakey") or self._hrakey is None:
            self._hrakey = self._fetch_datakey_(['RA_ICRS','RA','ra'], safeout=True)
        return self._hrakey

    @property
    def dec(self):
        """ declination coordinate, shortcut of self.get_dec()"""        
        return self.get_dec()

    @property
    def _deckey(self):
        """ The Declination key """
        if not hasattr(self,"_hdeckey") or self._hdeckey is None:
            self._hdeckey = self._fetch_datakey_(['DE_ICRS','DE','de','DEC','dec'], safeout=True)
        return self._hdeckey
    
    @property
    def skycoord(self):
        """ astropy SkyCoord of ra and dec """
        return self.get_skycoord()

    
class CatalogCollection( Catalog ):

    def __init__(self, catalogs=None, load_data=True):
        """ """
        super().__init__()
        if catalogs is not None:
            self.set_catalogs( catalogs, load_data=load_data)


    @classmethod
    def load(cls, filenames, names=None, squeeze=True, **kwargs):
        """ """
        filenames = np.atleast_1d(filenames)
        if names is None:
            names = [f"cat{i}" for i in range(len(filenames))]
            
        if len(filenames)==1 and squeeze:
            return Catalog.load(filenames[0], name=names[0], **kwargs)

        if len(names) != len(filenames):
            raise ValueError(f"names must have the same size as filenames {len(names)} vs. {len(filenames)}")
        
        return cls([Catalog.load(filename_, name=name_, **kwargs)
                    for filename_, name_ in zip(filenames, names)])
    

    @classmethod        
    def read_cvs(cls, filenames, name="catalog", index_col=None, readprop={}, **kwargs):
        """ """
        return NotImplementedError("read_cvs to be implemented")

    @classmethod
    def read_fits(cls, filename, dataext=1, headerext=None, name="catalog",
                      index_col='Source', **kwargs):
        """ """
        return NotImplementedError("read_fits to be implemented")

    def read_psfcat(cls, psfcat, name="ztfcat"):
        """ loads from ziff.ztfcat[0] """
        return NotImplementedError("read_fits to be implemented")


    def get_as_xyformat(self, xyformat, filtered=False, **kwargs):
        """ """
        newcoll = self.__class__( self._call_down_("get_as_xyformat", isfunc=True,
                                                xyformat=xyformat, filtered=filtered, *kwargs),
                                     load_data=True)
        newcoll.change_name(f"{xyformat}format_"+ ("filtered_" if filtered else "") +self.name)
        return newcoll
        
        
    # ----- #
    #  I/O  #
    # ----- #
    def write_to(self, savefile, filtered=True, extension=None, overwrite=True,
                     safeexit=False, **kwargs):
        """ """
        if len(np.atleast_1d(savefile)) == self.ncatalogs:
            self._call_down_("write_to", isfunc=True,
                              enumargs=savefile, # this will be go in one at the time.
                              filtered=filtered, extension=extension,
                              overwrite=overwrite, safeexit=safeexit, **kwargs)
        else:
            return NotImplementedError("single write_to not implemented for CatalogCollection")

    def to_fits(self):
        """ """
        print("not done yet")
    def to_csv(self):
        """ """
        print("not done yet")
        
    # ================ #
    #   Methods        #
    # ================ #
    def _multi_set_(self, key, value, setkey=None):
        """ this do self.{key} = [value] """
        value = np.atleast_1d(value)
        if len(value) == 1:
            value = [value[0]]*self.ncatalogs
                
        if len(value) != self.ncatalogs:
            raise ValueError(f"size of {key} must be one or one per catalogs {self.ncatalogs}. {len(value)} given. ")

        if setkey is None:
            setkey = f"_{key}"

        setattr(self, setkey, value)

    def _multi_set_down_(self, key, values, **kwargs):
        """ this loops of over the catalog to call catalog.set_{key}(value, **kwargs) """
        values = np.atleast_1d(values)
        if len(values) == 1:
            values = [values[0]]*self.ncatalogs
                
        if len(values) != self.ncatalogs:
            raise ValueError(f"size of {key} must be one or one per catalogs {self.ncatalogs}. {len(value)} given. ")
        
        return [getattr(cat,f"set_{key}")(v_, **kwargs)
                    for (cat,v_) in zip(self.catalogs, values)]

    def _call_down_(self, key_, enumargs=None, isfunc=True, **kwargs):
        """ this loops of over the catalogs to call catalog.{key}(*args, **kwargs) if isfunc or catalog.{key} if not """
        if not isfunc:
            return [getattr(c_, key_) for c_ in self.catalogs]

        if enumargs is None:
            return [getattr(c_, key_)(**kwargs) for c_ in self.catalogs]

        out = []
        for i, c_ in enumerate(self.catalogs):
            if len(np.atleast_1d(enumargs[i])) == 1:
                out.append(getattr(c_, key_)(enumargs[i], **kwargs))
            else:
                out.append(getattr(c_, key_)(*enumargs[i], **kwargs))
        return out

    # -------- #
    # LAODER   #
    # -------- #
    def _load_data_(self, names="default"):
        """ """
        dataframes = self._call_down_("data", isfunc=False)
        if names in ["default"]:
            names = self._call_down_("name", isfunc=False)
            
        self.set_data(dataframes, keys=names)

    def build_filename(self, prefix="", extension=".fits", percatalog=True, **kwargs):
        """ returns prefix+self.name+extension """
        if len( np.atleast_1d(prefix) ) == 1 and not percatalog:
            prefix = np.atleast_1d(prefix)[0]
            return super().build_filename(prefix, extension=extension, **kwargs)
        
        if len( np.atleast_1d(prefix) ) == 1: # same for all
            prefix = [np.atleast_1d(prefix)[0]]*self.ncatalogs

        if not extension.startswith("."):
            extension = f".{extension}"
            
        return [f"{prefix_}{self.name}{i}{extension}" for i,prefix_ in enumerate(prefix)]

    
    def build_sky_from_bkgdimg(self, bkgdimg, stampsize, askey="sky",
                                   npfunc="nanmean", show_progress=True,
                                   update=True):
        """ """
        if len(np.shape(bkgdimg))==2:
            bkgdimg = [bkgdimg]
            
        unique_bkgdimg = np.shape(bkgdimg)[0]==1
        stampsize = np.atleast_1d(stampsize)
        unique_stamp = len(stampsize)==1
        
        out = [c.build_sky_from_bkgdimg(bkgdimg[0] if unique_bkgdimg else bkgdimg[i],
                                        stampsize[0] if unique_stamp else stampsize[i])
                                      for i,c in enumerate(self.catalogs)
                ]
    
        if update:
            self._load_data_()
        return out
    
    def build_mask_from_maskimg(self, maskimg, stampsize, update=True):
        """ """
        if len(np.shape(maskimg))==2:
            maskimg = [maskimg]
            
        unique_masking = np.shape(maskimg)[0]==1
        stampsize = np.atleast_1d(stampsize)
        unique_stamp = len(stampsize)==1
        
        out =  [c.build_mask_from_maskimg(maskimg[0] if unique_masking else maskimg[i],
                                          stampsize[0] if unique_stamp else stampsize[i])
                                    for i,c in enumerate(self.catalogs)
                ]
        if update:
            self._load_data_()
        return out
        
    def load_xy_from_radec(self, update=True, overwrite=False, returns=False):
        """ """
        out =  self._call_down_("load_xy_from_radec", isfunc=True, update=True,
                                    overwrite=False, returns=False)
        if update:
            self._load_data_()
        return out
    
    def load_radec_from_xy(self, update=True, overwrite=False, returns=False):
        """ """
        out = self._call_down_("load_radec_from_xy", isfunc=True, update=True,
                                   overwrite=False, returns=False)
        if update:
            self._load_data_()
            
        return out

    def _guess_xyformat_(self):
        """ """
        return self._call_down_("_guess_xyformat_", isfunc=True)
    
    # -------- #
    # SETTER   #
    # -------- #
    def set_catalogs(self, catalogs, load_data=True, force_xyformat=None):
        """ """
        if force_xyformat is None:
            self._catalogs = np.atleast_1d(catalogs)
        else:
            self._catalogs = [c_ if c_._xyformat == force_xyformat else c_.get_as_xyformat(force_xyformat)
                                  for c_ in np.atleast_1d(catalogs)]
            
        if load_data:
            self._load_data_()


    def _clean_format_(self, force_xyformat="numpy"):
        """ """
        self.set_catalogs(self.catalogs, force_xyformat=force_xyformat)
        
    # change_name OK
    def set_data(self, dataframes, keys=None, clean_nameduplicate=True):
        """ """
        if keys is None:
            keys = [f"cat{i}" for i in range(self.ncatalogs)]

        # SingleIndex
        if len(dataframes) == self.ncatalogs:
            if clean_nameduplicate:
                keys = avoid_duplicate(keys)
            
            dataframes = pandas.concat(dataframes, keys=keys)
            
        elif type(dataframes.index) is not pandas.MultiIndex:
            raise TypeError("dataframes must be MultiIndex or list of dataframes")
        
        return super().set_data(dataframes)
        
    def set_wcs(self, wcs):
        """ """
        self._multi_set_down_("wcs", wcs)

    def set_header(self, header):
        """ """
        self._multi_set_down_("header", header)
    
    def set_mask(self, mask):
        """ """
        self._multi_set_down_("mask", mask)

    # -------- #
    #  GETTER  #
    # -------- #

    def get_filtered(self, **kwargs):
        """ """
        filtered_cat = [cat.get_filtered(**kwargs) for cat in self.catalogs]
        return self.__class__(catalogs=filtered_cat)

    def get_data(self, catname=None, source=None, filtered=False, as_hdu=False):
        """

        Parameters
        ----------
        catname: [string]
            Get the only the data assocated to the given catalog

        source: [string]
            Get only the data corresponding to the given source
            = ignored if catname is not None =
            
        filtered: [bool]
            Do you want the filtered data ?
        

        as_hdu: [bool]
            shall the data be returned in fits.hdu ?
    
        Returns
        -------
        DataFrame (or hdu, see as_hdu)
        """
        if not self.has_data():
            raise AttributeError("No data set yet. Use self.set_data()")

        if catname is not None:
            d_ = self.get_subdata(catname, level=0, filtered=filtered)
        elif source is not None:
            d_ = self.get_subdata(source, level=1, filtered=filtered)
        else:
            d_ = self.data[~self.filterout] if filtered else self.data
            
        if as_hdu:
            return dataframe_to_hdu(d_)
        return d_
    
    def get_subdata(self, index, level, filtered=False):
        """ """
        if filtered:
            return self.data[~self.filterout].xs(index, level=level)
            
        return self.data.xs(index, level=level)

    
    def get_config(self, first=True):
        """ returns the current configuration """
        config_list = self._call_down_("get_config", isfunc=True)
        if first:
            return config_list[0]
        return config_list

    def measure_isolation(self, refcat=None, seplimit=8):
        """ 
        
        Parameters
        ----------
        refcat: [Catalog or None] -optional-
            Refence catalog. Isolation will be measured as matching of self with this catalog.
            = IMPORTANT, refcat must include at least *all* the current catalog entries =
            If None, self will be used (self isolation).

        seplimit: [float] -optional-
            isolation distance in arcsec
        
        Returns
        -------
        
        """
        self._call_down_("measure_isolation", refcat=refcat, seplimit=seplimit, isfunc=True)
        self._load_data_()
        
    #---------- #
    # FILTERING #
    #---------- #
    def update_filter(self):
        """ """
        self._call_down_("update_filter", isfunc=True)
        self._load_data_()
    
    def add_filter(self, key, range_values, name = None, update=True):
        """ """
        self._call_down_("add_filter", key=key, range_values=range_values,
                             name=name, isfunc=True, update=False)
        if update:
            self.update_filter()

    def remove_filter(self, name):
        """ """
        if name in self.data.keys():
            self.data.drop(name, axis=1, inplace = True)
            self._filters.pop(name)
            self.update_filter()
        else:
            raise ValueError(f"Filter {name} not found in dataframe.")
    # ================ #
    #   Property       #
    # ================ #
    @property
    def catalogs(self):
        """ list of ziff.Catalog"""
        return self._catalogs

    @property
    def catnames(self):
        """ """
        return self.data.index.get_level_values(0).unique()
    
    def has_catalogs(self):
        """ """
        return hasattr(self,"_catalogs") and self._catalogs is not None and len(self._catalogs)>0

    @property
    def ncatalogs(self):
        """ """
        return len(self.catalogs)

    # - super it
    @property
    def data(self):
        """ """
        if not hasattr(self, "_data"):
            self._load_data_()
            
        return super().data

    @property
    def wcs(self):
        """ """
        return self._call_down_("wcs", isfunc=False)

    def has_wcs(self, test="all"):
        """ """
        return getattr(np, test)(self._call_down_("has_wcs", isfunc=True))

    @property
    def mask(self):
        """ """
        return np.asarray( self._call_down_("mask", isfunc=False) )

    # Fix the xyformat is needed
    def _guess_xyformat_(self):
        """ """
        formats = np.asarray( self._call_down_("xyformat", isfunc=False) )
        if len(np.unique(formats)) == 1:
            print(f"setting _xyformat to {np.unique(formats)[0]}")
            self._xyformat = np.unique(formats)[0]
        else:
            warnings.warn("Several xyformat. Cleaning the catalog to get all in 'numpy'")
            self._clean_format_(force_xyformat="numpy")
            self._xyformat = "nuympy"

            
#    @property
#    def filterout(self):
#        """ """
#        return np.asarray( self._call_down_("filterout", isfunc=False) )
    
    @property
    def header(self):
        """ """
        return self._call_down_("header", isfunc=False)
    
######################
#                    #
#  Derived Catalogs  #
#                    #
######################

class _CatalogHolder_( object ):
    """ """
    
    # ================ #
    #   Methods        #
    # ================ #
    # -------- #
    #  I/O     #
    # -------- #
    def load_catalog(self, filename, name, **kwargs):
        """ """
        self.set_catalog(Catalog.load(filename, name=name, multi_ok=True,
                            **{**{"extension":1},**kwargs}, ))
                
    # -------- #
    #  SETTER  #
    # -------- #
    def set_catalog(self, catalog, name=None):
        """ Add a new catalog """
        if name is None:
            name = catalog.name
            
        self.catalog[name] = catalog

    # -------- #
    #  GETTER  #
    # -------- #
    def get_catalog(self, catalog, chipnum=None, xyformat=None,
                        filtered=False, shuffled=False,
                        add_filter=None, writeto=None, writetoprop={}):
        """ Eval if catalog is a name or an object. Returns the object 
        = This returns a copy of the requested catalog = 
        

        add_filter: [None or dict] -optional-
            filter_format: {name: [key, vrange]}
            like: add_filter={'gmag_outrange':['gmag', [15,19]]}
            This filter will be added to the returned catalog

        
        """
        #
        # A bit messy but clean the issue of list of catalog if any.
        #
        if isinstance(catalog, str):
            catalog = self.catalog[catalog]
        
        if chipnum is not None and CatalogCollection in catalog.__class__.__mro__:
            catalog = catalog.catalogs[chipnum]

        # Build a copy
        catalog = catalog.get_catalog()

        #
        # - Add filter if any
        if add_filter is not None:
            for fname, values in add_filter.items():
                key, vrange = values
                catalog.add_filter(key, vrange, name=fname)
        #
        # - Change format or
        if xyformat is not None or filtered or shuffled:
            # This is a copy
            catalog = catalog.get_catalog(filtered=filtered, shuffled=shuffled, xyformat=xyformat)

        if writeto is not None:
            #
            # - Special cases
            if writeto == "default":
                writeto = catalog.build_filename(self.prefix)
            elif  writeto == "shape":
                writeto = catalog.build_filename(self.prefix+"shapecat_", extension=".fits")
            elif  writeto == "prefixtmp":
                writeto = catalog.build_filename(self.prefix+"tmpcat_", extension=".fits")
            elif  writeto == "tmp":
                writeto = catalog.build_filename("tmpcat_", extension=".fits")
            
                
            catalog.write_to(writeto, store_filename=True, #stores as catalog.filename
                                 **{**dict(filtered=False), **writetoprop})
            
        return catalog
        
    def _get_stored_catalog_(self, catalog, chipnum=None, fileout=None,
                                 filtered=True, shuffled=False, xyformat=None,
                                 add_filter=None, overwrite=True):
        """ """
        print("DEPRECATED, use get_catalog(writeto=)")
        
        cat = self.get_catalog(catalog, chipnum=chipnum, xyformat=xyformat,
                                shuffled=shuffled, filtered=filtered,
                                add_filter=add_filter)

        if cat.npoints == 0:
            warnings.warn("no data in the given catalog to store.")
            return cat, "None"
        
        if fileout is None or fileout in ["default"]:
            fileout = cat.build_filename(self.prefix)
        elif fileout in ["tmp"]:
            fileout = cat.build_filename("tmpcat_", extension=".fits")
            
        #else -> fileout = fileout
        
        cat.write_to(fileout, overwrite=overwrite)
        return cat, fileout
    
    def get_stacked_cat_df(self):
        """ """        
        dfs = {}
        for cat in self.catalog:
            c = self.get_catalog(cat)
            df = []
            for i in range(self.nimgs):
                dfi = c[i].data
                dfi = dfi.loc[~dfi['filterout']]
                df.append(dfi)
            dfs[cat] = pandas.concat(df)
        return dfs
    
    # ================ #
    #   Properties     #
    # ================ #
    # - Catalogs
    @property
    def catalog(self):
        """ Dictionnary of catalogs """
        if not hasattr(self, "_catalog"):
            self._catalog = {}
            
        return self._catalog
    
    def has_catalog(self):
        """ """
        return hasattr(self,"_catalog") and len(self.catalog)>0
    
# End of catalog.py ========================================================

#  LocalWords:  toprimary

