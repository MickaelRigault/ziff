
""" PSF Shape Analysis """

import os
import warnings
import numpy as np
import pandas

import dask
import dask.dataframe as dd
from dask import distributed

from ztfquery import buildurl,io

from .. import base
from .. import io as zio
from . import basecluster
from .psf import get_ziff_psf_cat

def compute_shapes(file_, use_dask=False, incl_residual=True, incl_stars=True,
                       whichpsf="psf_PixelGrid_BasisPolynomial5.piff",
                       stamp_size=17):
    """ high level script function of ziff to 
    - compute and store the stars and psf-model shape parameters

    This re-perform the last steps of ziffit_single.

    = Dask oriented =

    """
    delayed = basecluster.get_delayed_func(use_dask=use_dask)

    input_needed = delayed(get_ziff_psf_cat)(file_, whichpsf=whichpsf)
    ziff, psf, cat_toshape = input_needed[0],input_needed[1],input_needed[2]
    
    shapes       = delayed(base.get_shapes)(ziff, psf, cat_toshape, store=True, stamp_size=stamp_size,
                                                incl_residual=incl_residual, incl_stars=incl_stars)
    
    return shapes[["sigma_model","sigma_data"]].median(axis=0).values

def build_digitalized_shape(filenames, urange, vrange, savefile, bins=200, chunks=300, 
                            minimal=True, **kwargs):
    """ high level script function of ziff to 
    - read the computed shape parameters

    = Dask oriented =

    - Returns delayed calls - 

    """
    filedf = zio.get_filedataframe(filenames)
    grouped = filedf.groupby("filefracday")
    groupkeys = list( grouped.groups.keys() )
    
    bins_u = np.linspace(*urange, bins)
    bins_v = np.linspace(*vrange, bins)

    chunck_filenames = [np.concatenate([grouped["filename"].get_group(g_).values for g_ in chunk])
                            for chunk in np.array_split(groupkeys, chunks)]
    
    delayed_chunks = []
    for i, cfile in enumerate(chunck_filenames):
        delayed_chunks.append(dask.delayed(get_sigma_data)(cfile, bins_u, bins_v, minimal=minimal,
                                                savefile=None if savefile is None else savefile.replace(".parquet",f"_chunk{i}.parquet"),**kwargs))
        
    return delayed_chunks
    

def build_digitalize_psfdata(filenames, valrange, key, savefile,
                              bins=200, chunks=300, **kwargs):
    """ """
    filedf = zio.get_filedataframe(filenames)
    grouped = filedf.groupby("filefracday")
    groupkeys = list( grouped.groups.keys() )
    
    bins_val = np.linspace(*valrange, bins)
    chunck_filenames = [np.concatenate([grouped["filename"].get_group(g_).values for g_ in chunk])
                            for chunk in np.array_split(groupkeys, chunks)]
    
    delayed_chunks = []
    for i, cfile in enumerate(chunck_filenames):
        delayed_chunks.append(dask.delayed(get_binned_data)(cfile, bins_val, key,
                                savefile=None if savefile is None else savefile.replace(".parquet",
                                                                        f"_{key}{bins}bins_chunk{i}.parquet"),**kwargs))

    return delayed_chunks

def get_binned_data(files, bin_val, key, savefile=None, columns=None,
                    quantity='sigma', normref="model"):
    """ """
    filefracday = [f.split("/")[-1].split("_")[1] for f in files]
    df = pandas.concat([pandas.read_parquet(f, columns=columns) for f in files], keys=filefracday
                           ).reset_index().rename({"level_0":"filefracday"}, axis=1)


    norm = df.groupby(["obsjd"])[f"{quantity}_{normref}"].transform("median")
    
    df[f"{quantity}_data_n"] = df[f"{quantity}_data"]/norm
    df[f"{quantity}_model_n"] = df[f"{quantity}_model"]/norm
    df[f"{quantity}_residual"] = (df[f"{quantity}_data"]-df[f"{quantity}_model"])/df[f"{quantity}_model"]
    
    df[f"{key}_digit"] = np.digitize(df[key], bin_val)
    if savefile:
        df.to_parquet(savefile)
        
    return df
    
def get_sigma_data(files, bins_u, bins_v,
                    minimal=False,
                   quantity='sigma', normref="model", incl_residual=True,
                   basecolumns=['u', 'v', 'ccdid', 'qid', 'rcid', 'obsjd', 'fieldid','filterid', 'maglim'],
                   savefile=None,
                  ):
    if minimal:
        shape_columns = [f"{quantity}_data",  f"{quantity}_model"]
        incl_residual = False
    else:
        shape_columns = [f"sigma_data",f"sigma_model"] + \
                        [f"shapeg2_data",f"shapeg2_model"] + \
                        [f"shapeg1_data",f"shapeg1_model"] + \
                        [f"centerv_data",f"centerv_model"] + \
                        [f"centeru_data",f"centeru_model"]
    columns = basecolumns + shape_columns
    if incl_residual: 
        columns += ["residual"]

    filefracday = [f.split("/")[-1].split("_")[1] for f in files]
    df = pandas.concat([pandas.read_parquet(f, columns=columns) for f in files], keys=filefracday
                           ).reset_index().rename({"level_0":"filefracday"}, axis=1)
    
    norm = df.groupby(["obsjd"])[f"{quantity}_{normref}"].transform("median")
    
    df[f"{quantity}_data_n"] = df[f"{quantity}_data"]/norm
    df[f"{quantity}_model_n"] = df[f"{quantity}_model"]/norm
    df[f"{quantity}_residual"] = (df[f"{quantity}_data"]-df[f"{quantity}_model"])/df[f"{quantity}_model"]
    df["u_digit"] = np.digitize(df["u"],bins_u)
    df["v_digit"] = np.digitize(df["v"],bins_v)
    if savefile:
        df.to_parquet(savefile)
    return df


def fetch_parquetsource_data(filename, datakey, sources=None):
    """ """
    data    = pandas.read_parquet(io.get_file(filename, suffix="psfshape.parquet", check_suffix=False),
                                    columns=[datakey])
    if sources is not None:
        data = data.loc[sources]
        
    return data.values.tolist()




# =========================== #
#                             #
#  PSF Shape Analysis Class   #
#                             #
# =========================== #

class PSFShapeAnalysis( basecluster.DaskCluster ):
    """ """
            
    @classmethod
    def from_directory(cls, directory, patern="*.parquet",
                        urange=None, vrange=None, bins=None,
                        client=None):
        """ provide the directory where the digitilized and chunked psfdata are.
        """
        this = cls(client=client)
        this.load_fromdir(directory, patern=patern,
                            urange=urange, vrange=vrange, bins=bins)
        return this

    @classmethod
    def from_pifffiles(cls, pifffiles, urange, vrange, bins, client,
                           psf_suffix=None, subdir=None, digit_basename="psfshape",
                           build_shapes=True, stamp_size=17, chunks=300):
        """ """
        print("This has not been test")
        this = cls(client=client)
        this.set_binning(urange=urange, vrange=vrange, bins=bins)
        if psf_suffix is None:
            psf_suffix = zio.parse_psf_suffix(pifffiles[0], expand=False)
            # psfmodel has the format psf_model_interp.piff
            
        #
        # - build shapes        
        if build_shapes:
            f_shapes = this.cbuild_shapes(pifffiles, psf_suffix=psfmodel, stamp_size=stamp_size)
            futures_, _ = distributed.wait(f_shapes)
            
        #
        # - To compute
        to_compute = [f.replace(psfmodel, "psfshape.parquet") for f in pifffiles
                          if os.path.isfile(f.replace(psfmodel, "psfshape.parquet"))]

        dirout = zio.get_digit_dir(psf_suffix, subdir)
        savefile_base = os.path.join(dirout, digit_basename)
        f_digit = this.cbuild_digitalized_shape(to_compute, savefile_base, chunks=chunks)
        # - 
        futures_, _ = distributed.wait(f_digit)
        # -
        bins = self.binning["bins"]
        this.load_fromdir(dirout, patern=savefile_base+f"*{bins}*.parquet")
        return this

    @classmethod
    def from_shapefiles(cls, shapefiles, urange, vrange, bins, client, chunks=300,
                            subdir=None, digit_basename="psfshape", **kwargs):
        """ """
        this = cls(client=client)
        this.set_binning(urange=urange, vrange=vrange, bins=bins)

        dirout = zio.get_digit_dir(subdir=subdir)
        savefile_base = os.path.join(dirout, digit_basename)
        this.cbuild_digitalized_shape(shapefiles, savefile_base,
                                           chunks=chunks, load_data=True, **kwargs)
        return this
        
    # --------- #
    #  GETTER   #
    # --------- #

        
    # --------- #
    #  Builder  #
    # --------- #
    def cbuild_shapes(self, sciimgfiles, client=None, psf_suffix="psf_PixelGrid_BasisPolynomial5.piff",
                          stamp_size=17, **kwargs):
        """ Measures the shapes parameters from the sciimgs (looking for the psf files) """
        delayed_shapes = [compute_shapes(f_, use_dask=True, incl_residual=True, incl_stars=True,
                                         whichpsf=psf_suffix,
                                         stamp_size=stamp_size, **kwargs)
                              for f_ in sciimgfiles]

        #
        # - Client or not
        #
        client = self.get_client(client=client)
        if client is None:
            warnings.warn("list of dask.delayed returned")
            return delayed_shapes
        
        # - Client, so let's compute   
        return client.compute(delayed_shapes)
        
    def cbuild_digitalized_shape(self, parquetfiles, savefile_base, client=None,
                                      chunks=300, load_data=False, **kwargs):
        """ digitalize the shapes files (parquetfiles) and store them per chunk.

        Parameters
        ----------
        parquetfiles: [list of path]
            The psfshape parquet files to be digitalized.w

        savefile_base: [string]
            Incomplete full path used to create the structure of the digitalized data.
            the finale names will be:
            savefile_base + "_{bins}bins_chunk{i}.parquet"
            

        **kwargs goes to build_digitalized_shape
        Returns
        -------
        list of: "futures or delayed" depending on client.
        """
        
        if len(parquetfiles) <= chunks:
            raise ValueError(f"more chunks than files ({chunks} vs. {len(parquetfiles)}")

        if np.any([v is None for v in self.binning.values()]):
            raise AttributeError(f"you need to set all the binning information: see self.set_binning(). current: {self.binning}")
        
        bins = self.binning['bins']
        savefile = savefile_base + f"_{bins}bins.parquet"
        data_delayed = build_digitalized_shape(parquetfiles, chunks=chunks,
                                                savefile=savefile, # _chunk{i} added inside 
                                                **{**self.binning, **kwargs})
        #
        # - Client or not
        #
        client = self.get_client(client=client)
        if client is None:
            warnings.warn("list of dask.delayed returned")
            return data_delayed
            
        # - Client, so let's compute
        futures = client.compute(data_delayed)
        if load_data:
            futures = distributed.wait(futures)
            self.load_fromdir(os.path.dirname(savefile), os.path.basename(savefile).replace(".parquet","*.parquet"),
                                  **self.binning)
            return
        
        return futures

    # --------- #
    #  SETTER   #
    # --------- #
    def set_client(self, client, persist=True):
        """ """
        self._client = client
        if self.has_data() and persist:
            self._data = self.client.persist(self.data)
            
    def set_data(self, data, urange=None, vrange=None, bins=None, persist=True):
        """ """
        self._data = data
        self.set_binning(urange=urange, vrange=vrange, bins=bins)
        if "u_digit,v_digit" not in self.data.columns:
            self.data["u_digit,v_digit"] = self.data["u_digit"].astype("str")+","+ self.data["v_digit"].astype("str")

        if persist and self.has_client():
            self._data = self.client.persist(self.data)

    def set_binning(self, urange, vrange, bins):
        """ """
        self._binning = {"urange":urange, "vrange":vrange, "bins":bins}

    # --------- #
    #  LOADER   #
    # --------- #
    def load_fromdir(self, directory, patern="*.parquet", urange=None, vrange=None, bins=None):
        """ provide the directory where the digitilized and chunked psfdata are.
        """
        data = dd.read_parquet( os.path.join(directory, patern) )
        self.set_data(data, urange=urange, vrange=vrange, bins=bins)
        
    def load_medianserie(self):
        """ """
        self._seriemedian = self.grouped_digit[["sigma_model_n","sigma_data_n","sigma_residual"]
                                              ].apply(pandas.Series.median).compute()

    def load_shapemaps(self):
        """ """
        hist2d = np.ones((4, self.binning["bins"],self.binning["bins"]))*np.NaN
        
        for k,v in self.seriemedian.iterrows():
            hist2d[0, k[1], k[0]] = v["sigma_data_n"]
            hist2d[1, k[1], k[0]] = v["sigma_model_n"]
            hist2d[2, k[1], k[0]] = v["sigma_residual"]

        for k,v in self.grouped_digit.size().compute().iteritems():
            hist2d[3, k[1], k[0]] = v

        self._shapemaps = {"data": hist2d[0],
                           "model": hist2d[1],
                           "residual": hist2d[2],
                           "density": hist2d[3]}
    # --------- #
    #  GETTER   #
    # --------- #
    def get_center_pixel(self):
        """ """
        ucenter = np.median(self.bins_u[~np.all(np.isnan(self.shapemaps["data"]), axis=0)])
        vcenter = np.median(self.bins_v[~np.all(np.isnan(self.shapemaps["data"]), axis=1)])
        return ucenter,vcenter

    def get_metapixels(self, resrange=None, modelrange=None, datarange=None, densityrange=None,
                           within=None, as_string=False):
        """ 
        Parameters
        ----------
        resrange, modelrange, datarange, density: [float, float]
            min value and max value to select the metapixels.
            Applied on residual, model or data map respetively
            e.g. resrange=[-0.1,0.1]
            

        within: [(float, float), float]
            u and v of the centroid + radius (in unit of u and v)
            e.g. within=((3000,-3000), 500)
                 within=(self.get_center_pixel(), 500)

        """
        flag = []
        for key,vrange in zip(
                        ["residual", "model", "data", "density"],
                        [resrange, modelrange, datarange, densityrange]):
            if vrange is not None:
                flag.append( (self.shapemaps[key]>vrange[0]) * (self.shapemaps[key]<vrange[1]))

        if len( flag ) >0:
            mpxl_args = np.argwhere(np.all(flag, axis=0))
        else:
            mpxl_args = None


        if within is not None:
            (ucentroid, vcentroid), dist_ = within
            if mpxl_args is None:
                ub = self.bins_u-ucentroid
                vb = self.bins_v-vcentroid
                mpxl_args = np.sqrt(ub**2+ vb**2)<dist_
            else:
                ub = self.bins_u[mpxl_args.T[1]]-ucentroid
                vb = self.bins_v[mpxl_args.T[0]]-vcentroid
                mpxl_args = mpxl_args[np.sqrt(ub**2+ vb**2)<dist_]

        if as_string:
             return np.asarray([f"{u_},{v_}" for u_,v_ in mpxl_args])
         
        return mpxl_args

    def get_fgroup_filename(self, metapixels, filenames=None, nfiles=None):
        """ get filedata grouped by filename and associated filenames (see options)

        Parameters
        ----------
        filenames: [list of path] -optional-
            You can limit the list of file that are going to be analysed.
            if None, all the files that have at least 1 target in the given metapixels
        
        nfiles: [int] -optional-
            only look at the first n-files. 
            = ignored if filenames is not None =
            
        Returns
        -------
        groupby(filename), filenames

        """
        fdata = self.get_metapixels_filedata(metapixels).compute()
        fgroup = fdata.groupby("filename")
        if filenames is None:
            filenames = list(fgroup.groups.keys())
            if nfiles is not None:
                filenames = filenames[:nfiles]

        return fgroup, filenames
    
    def get_metapixels_filedata(self, metapixels):
        """ """
        subdata = self.data[ ["Source","filefracday","fieldid","ccdid","qid","filterid"]
                           ][ self.data['u_digit,v_digit'].isin(metapixels) ]
        subdata["filename"] = buildurl.build_filename_from_dataframe(subdata)
        return subdata[["filename","Source"]]
        
    def get_metapixel_data(self, metapixel, columns=None):
        """ """
        if columns is not None:
            self.data[columns][self.data['u_digit,v_digit'].isin(metapixel)]
        else:
            self.data[self.data['u_digit,v_digit'].isin(metapixel)]
        return data

    def fetch_metapixel_data(self, metapixel, datakey):
        """ """
        subdata = self.get_metapixel_data(metapixel, columns=["Source","filefracday","fieldid","ccdid","qid","filterid"])
        subdata["filename"] = buildurl.build_filename_from_dataframe(subdata)
        fdata = subdata[["filename","Source"]]
        
    def get_metapixel_sources(self, metapixel, columns=["filename", "Source"], compute=True):
        """ """
        metapixeldata = self.grouped_digit.get_group(tuple(metapixel))
        metapixeldata["filename"] = buildurl.build_filename_from_dataframe(metapixeldata)
        if columns is not None and compute:
            return metapixeldata[columns].compute()
        
        return metapixeldata

    # ------------- #
    # Client GETTER #
    # ------------- #
    def cget_median_stamp(self, metapixels, client=None, on="stars", gather=True):
        """ 
        Parameters
        ----------
        on: [string] -optional-
            on could be stars or residual
           
        """
        print("TO BE TESTED TO CHECK IF DEPRECATED, SEE cget_metapixels_data")
        client = self.get_client(client)
        if client is None:
            raise ValueError("client requested for this method.")

        
        # dmetapixeldata is lazy
        all_meta = []
        for p_ in metapixels:
            metapixeldata = self.grouped_digit[["filefracday","fieldid","ccdid","qid","filterid","Source"]
                                                   ].get_group(tuple(metapixel))
            metapixeldata["filename"] = buildurl.build_filename_from_dataframe(metapixeldata)
        
        dmetapixeldata = [self.get_metapixel_sources(l_, compute=False)
                              for l_ in metapixels]
        # all metapixeldata are computed but still distribution inside the cluster
        # they are 'futures'
        #  They are computed together for the share the same data files
        f_metapixeldata  = client.compute(dmetapixeldata)
        
        # Logic. Then work on the distributed data
        
        # Grab all the stars for each of them. Computation made on the respective cluster's computer
        f_stamps = client.map(fetch_parquetsource_data, f_metapixeldata, datakey=on)
            
        if gather:
            return client.gather(f_stamps)
        
        return f_stamps

    def cget_metapixels_data(self, metapixels, datakey, client=None, filenames=None, nfiles=None):
        """ 
        Parameters
        ----------
        client: [Dask Client]
            Dask client used for the computation.

        datakey: [string]
            Any in from the psfshape.parquet file.

        filenames: [list of path] -optional-
            You can limit the list of file that are going to be analysed.
            if None, all the files that have at least 1 target in the given metapixels
        
        nfiles: [int] -optional-
            only look at the first n-files. 
            = ignored if filenames is not None =

        """        
        fgroup, filenames = self.get_fgroup_filename(metapixels, filenames=filenames, nfiles=nfiles)
            
        d_data = [dask.delayed(fetch_parquetsource_data)(fname, sources=fgroup.get_group(fname).Source.values,
                                                        datakey=datakey) for fname in filenames]

        client = self.get_client(client=client)
        if client is None:
            warnings.warn("list of dask.delayed returned")
            return d_data
        
        return client.compute(d_data)
    
    # --------- #
    #  PLOTTER  #
    # --------- #
    def show_psfshape_maps(self, savefile=None, vmin="3", vmax="97", cvmin=-0.8, cvmax=0.8):
        """ """
        from ..plots import get_threeplot_axes, vminvmax_parser, display_binned2d

        fig, [axd, axm, axr], [cax, caxr] = get_threeplot_axes(fig=None, bottom=0.1, hxspan=0.09)


        vmin, vmax = vminvmax_parser(self.shapemaps["model"][self.shapemaps["model"]==self.shapemaps["model"]], vmin, vmax)

        prop = dict(xbins=self.bins_u, ybins=self.bins_v, transpose=False, 
                    vmin=vmin, vmax=vmax, cmap="coolwarm")

        imd = display_binned2d(axd, self.shapemaps["data"], **prop)
        # -> Model    
        imm = display_binned2d(axm, self.shapemaps["model"], cax=cax, **prop)

        imr = display_binned2d(axr, self.shapemaps["residual"]*100, cax=caxr, 
                                      **{**prop,**{"cmap":"coolwarm",
                                                   "vmin":cvmin,"vmax":cvmax}})
        imr.colorbar.set_ticks([np.round(-cvmin*2/3, 1), 0, np.round(cvmax*2/3, 1)])
        [ax.set_yticklabels(["" for _ in ax.get_yticklabels()]) for ax in [axm, axr]]


        textprop = dict(fontsize="small", color="0.3", loc="left")
        axd.set_title("data", **textprop)
        axm.set_title("model", **textprop)
        axr.set_title("(data-model)/model [%]", **textprop)
        fig.text(0.5, 0.99, "PSF width (normed per exposure)", va="top", ha="center", weight="bold")
        
        if savefile:
            fig.savefig(savefile, dpi=300)
            
        return fig

    # ================= #
    #    Properties     #
    # ================= #
    @property
    def data(self):
        """ """
        return self._data

    def has_data(self):
        """ """
        return hasattr(self, "_data") and self._data is not None

    @property
    def grouped_digit(self):
        """ """
        if not hasattr(self,"_grouped_digit") or self._grouped_digit is None:
            self._grouped_digit = self.data.groupby(["u_digit","v_digit"])
            
        return self._grouped_digit

    @property
    def seriemedian(self):
        """ """
        if not hasattr(self,"_seriemedian") or self._seriemedian is None:
            self.load_medianserie()
            
        return self._seriemedian
    
    @property
    def shapemaps(self):
        """ """
        if not hasattr(self,"_shapemaps") or self._shapemaps is None:
            self.load_shapemaps()
            
        return self._shapemaps
    
    @property
    def binning(self):
        """ """
        return self._binning
    
    @property
    def bins_u(self):
        """ """
        return np.linspace(*self.binning["urange"], self.binning["bins"])

    @property
    def bins_v(self):
        """ """
        return np.linspace(*self.binning["vrange"], self.binning["bins"])
