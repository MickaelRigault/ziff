
import numpy as np
import pandas
from . import ziffit
from ztfquery import buildurl


def _get_metapixeldata_(grouped_digit, metapixel):
    """ """
    metapixeldata = grouped_digit.get_group(tuple(metapixel))
    metapixeldata["filename"] = buildurl.build_filename_from_dataframe(metapixeldata))
    return metapixeldata

def _fetch_residuals_(fgroups, datakey):
    datas = []
    for filename in list(fgroups.groups.keys())[:100]:
        source = fgroups.get_group(filename)["Source"].values
        fdata  = pandas.read_parquet(io.get_file(filename, suffix="psfshape.parquet", check_suffix=False),
                   columns=[datakey]).loc[source].values.tolist()

        datas.append(fdata)
    return np.squeeze(np.concatenate(datas))

def fetch_metapixel_psfshape_data(grouped_digit, metapixel, datakey, use_dask=False):
    delayed = ziffit._not_delayed_ if not use_dask else dask.delayed
    
    mdata   = delayed(_get_metapixeldata_)(grouped_digit, metapixel)
    fgroups = mdata.groupby("filename")
    datas = delayed(_fetch_residuals_)(fgroups, datakey)
    return datas

def _residual_to_skydata_(residuals, buffer=2):
    """ """
    residuals  = residuals.reshape(len(residuals), 15, 15)
    residuals[:, buffer:-buffer,buffer:-buffer] = np.NaN
    median_sky = np.median(residuals, axis=0)
    means      = np.nanmean(residuals, axis=(1,2))
    stds       = np.nanstd(residuals, axis=(1,2))
    npoints    =  np.sum(~np.isnan(residuals), axis=(1,2))
    return median_sky,[means, stds, npoints]

def fetch_metapixel_sky(grouped_digit, metapixel, use_dask=False, buffer=2):
    delayed   = ziffit._not_delayed_ if not use_dask else dask.delayed
    residuals = fetch_metapixel_psfshape_data(grouped_digit, metapixel, datakey="residual", use_dask=False)
    dataout   = delayed(_residual_to_skydata_)(residuals, buffer=buffer)
    return dataout




class PSFShapeAnalysis( object ):
    """ """
    def __init__(self):
        """ """

    @classmethod
    def from_directory(cls, directory, patern="*.parquet", urange=None, vrange=None, bins=None):
        """ """
        import os
        this = cls()
        data = dd.read_parquet(os.path.join(directory,patern))
        this.set_data(data, urange=urange, vrange=vrange, bins=bins)
        return this
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, data, urange=None, vrange=None, bins=None):
        """ """
        self._data = data
        self.set_binning(urange=urange, vrange=vrange, bins=bins)

    def set_binning(self, urange, vrange, bins):
        """ """
        self._binning = {"urange":urange, "vrange":vrange, "bins":bins}

    # --------- #
    #  LOADER   #
    # --------- #
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
                           within=None):
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
            (ucentroid, vcentroid), dist_ = self.get_center_pixel(), 2000
            if mpxl_args is None:
                ub = self.bins_u-ucentroid
                vb = self.bins_v-vcentroid
                mpxl_args = np.sqrt(ub**2+ vb**2)<dist_
            else:
                ub = self.bins_u[mpxl_args.T[1]]-ucentroid
                vb = self.bins_v[mpxl_args.T[0]]-vcentroid
                mpxl_args = mpxl_args[np.sqrt(ub**2+ vb**2)<dist_]

        return mpxl_args


    def get_metapixel_sources(self, metapixel, columns=["filename", "Source"], compute=True):
        """ """
        metapixeldata = self.grouped_digit.get_group(tuple(metapixel))
        metapixeldata["filename"] = buildurl.build_filename_from_dataframe(metapixeldata)
        if columns is not None and compute
            return metapixeldata[columns].compute()
        
        return metapixeldata

    # --------- #
    #  PLOTTER  #
    # --------- #
    def show_psfshape_maps(self, savefile=None, vmin="3", vmax="97"):
        """ """
        from ziff.plots import get_threeplot_axes, vminvmax_parser, display_binned2d

        fig, [axd, axm, axr], [cax, caxr] = get_threeplot_axes(fig=None, bottom=0.1, hxspan=0.09)


        vmin, vmax = vminvmax_parser(self.shapemaps["model"][self.shapemaps["model"]==self.shapemaps["model"]], vmin, vmax)

        prop = dict(xbins=self.bins_u, ybins=self.bins_v, transpose=False, 
                    vmin=vmin, vmax=vmax, cmap="coolwarm")

        imd = display_binned2d(axd, self.shapemaps["data"], **prop)
        # -> Model    
        imm = display_binned2d(axm, self.shapemaps["model"], cax=cax, **prop)

        imr = display_binned2d(axr, self.shapemaps["residual"]*100, cax=caxr, 
                                      **{**prop,**{"cmap":"coolwarm",
                                                   "vmin":-0.8,"vmax":+0.8}})
        imr.colorbar.set_ticks([-0.5,0,0.5])
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
