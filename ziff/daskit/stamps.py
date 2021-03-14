

""" NOT READY YET """

def _fetch_filesource_sky_(filename, buffer=2, datakey="stars", sources=None,
                               statistic="median", **kwargs):
    """ """
    res = _fetch_filesource_data_(filename, datakey, sources=sources)
    return _residual_to_skydata_(np.asarray(res), buffer=buffer, statistic=statistic, **kwargs)

def _fetch_filesource_data_(filename, datakey, sources=None):
    """ """
    data    = pandas.read_parquet(io.get_file(filename, suffix="psfshape.parquet", check_suffix=False),
                                    columns=[datakey])
    if sources is not None:
        data = data.loc[sources]
        
    return data.values.tolist()
    
def _residual_to_skydata_(residuals, buffer=2, statistic="median", returns="all"):
    """ 
    returns: [string]
       - stamp: the stamp (or list of see, statistic)
       - meanstat: the mean statistics (means, stds and npoints)
       - all: stamp, meanstat
    """
    if returns not in ["stamp", "meanstat", "all", "both", "*"]:
        raise ValueError(f"returns must be 'stamp', 'meanstat', 'all'/'both'/'*' {returns} given.")
    
    residuals  = residuals.reshape(len(residuals), 15, 15)
    residuals[:, buffer:-buffer,buffer:-buffer] = np.NaN
    if statistic is not None:
        stampout = getattr(np,statistic)(residuals, axis=0)
    else:
        stampout = residuals
    if returns == "stamp":
        return stampout
    means      = np.nanmean(residuals, axis=(1,2))
    stds       = np.nanstd(residuals, axis=(1,2))
    npoints    =  np.sum(~np.isnan(residuals), axis=(1,2))
    
    if returns == "meanstat":
        return [means, stds, npoints]
    else:
        return residuals, [means, stds, npoints]


