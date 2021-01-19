""" Wrapper of the piff.Star  """
import numpy as np
import pandas 

def get_star_psfmodel(star, psf, asarray=False, modeldraw=False, basemodel=False, fit_center=True):
    """ 
    
    Parameters
    ----------
    asarray: [bool] -optional-
        Format of the result, 
        - False: piff.Star
        - True: numpy array (piff.Star.image.array)

    basemodel: [bool] -optional-
        do you want the model stamp (i.e. sum(array)=1) (basemodel=True) 
        or the actual star model (including flux)

    fit_center: [bool] -optional-
        if you reflux, should the center be refit ?
        = ignored if basemodel is True =

    Returns
    -------
    piff.Star or 2d stamp array
    """
    #
    # - Single Case
    target_star = psf.interpolateStar( psf.model.initialize( star ) )
    if basemodel:
        new_star = psf.drawStar( target_star )
    elif modeldraw:
        new_star = psf.model.draw( target_star )
    else:
        new_star = psf.drawStar( psf.model.reflux(target_star, fit_center=fit_center) )
            
    return new_star.image.array if asarray else new_star

def stars_to_array(star, origin="image"):
    """ """
    return np.asarray([getattr(star_, origin).array for star_ in np.atleast_1d(star)])


def show_psfmodeling(star, modelstar, axes=None, title="", tight_layout=True, **kwargs):
    """ """
    import matplotlib.pyplot  as mpl
    #
    # - Setting figure
    if axes is None:
        fig = mpl.figure(figsize=[8,3])
        axd = fig.add_subplot(131)
        axm = fig.add_subplot(132)
        axr = fig.add_subplot(133)
    else:
        axd, axm, axr = axes
        fig = axd.figure
    # - 
    #

    prop = dict(origin="lower", cmap="cividis")
    proptext = dict(fontsize="medium", color="0.3", loc="left")
    
    # - Data
    axd.imshow(star.image.array, **prop)
    axd.set_title(f"data: ({star.x:.1f}, {star.y:.1f}) ", **proptext)
    axd.set_xticks([])
    axd.set_yticks([])

    # - Model
    axm.imshow(modelstar.image.array, **prop)
    axm.set_title("model", **proptext)
    axm.set_xticks([])
    axm.set_yticks([])
        
    # - Residual: Data-Model
    axr.imshow(star.image.array-modelstar.image.array, **prop)
    axr.set_title("data-model", **proptext)
    axr.set_xticks([])
    axr.set_yticks([])
    
    if tight_layout:
        fig.tight_layout()
        
    return fig

def vminvmax_parser(data_, vmin, vmax):
    """ """
    if vmin is None:
        vmin="0"
    if vmax is None:
        vmax = "100"
    if type(vmin) == str:
        vmin = np.percentile(data_, float(vmin))
    if type(vmax) == str:
        vmax = np.percentile(data_, float(vmax))
        
    return vmin, vmax

class StarCollection( object ):
    """ """
    def __init__(self, stars):
        """ """
        self.set_stars(stars)

    # --------- #
    #  SETTER   #
    # --------- #
    def set_stars(self, stars):
        """ """
        self._stars = np.atleast_1d(stars)
        self._starsarray = stars_to_array(self.stars, origin="image")
        self._u,self._v = np.asarray([[s_.u, s_.v] for s_ in self.stars]).T
                              
    def measure_psfmodel(self, psf, modeldraw=False, basemodel=False, fit_center=False):
        """ """
        self._psfmodel = [get_star_psfmodel(star_, psf, modeldraw=modeldraw,
                                                basemodel=basemodel, fit_center=fit_center, asarray=False)
                              for star_ in self.stars]
        self._psfmodelarray = stars_to_array(self.psfmodel, origin="image")

    def measure_shapes(self, which=["stars","psfmodel"], normalisation="nanmedian"):
        """ """
        for which in np.atleast_1d(which):
            shapes = {'flux': [], 'sigma': [], 
                      'shape_g1': [], 'shape_g2': [],
                      'flagout': [],"u":[],"v":[],"x":[],"y":[],
                      'center_u' : [],'center_v' : [],
                      'center_x' : [],'center_y' : []}
            
            for s_ in getattr(self, which):
                s_.run_hsm()
                flux, center_x, center_y, sigma, shape_g1, shape_g2, flag = s_.run_hsm()
                
                shapes['flux'].append(flux)
                #
                shapes['sigma'].append(sigma)
                #
                shapes['shape_g1'].append(shape_g1)
                shapes['shape_g2'].append(shape_g2)
                shapes['u'].append(s_.u)
                shapes['v'].append(s_.v)
                shapes['x'].append(s_.x)
                shapes['y'].append(s_.y)
                #
                shapes['center_x'].append(center_x)
                shapes['center_y'].append(center_y)
                shapes['center_u'].append(s_.center[0])
                shapes['center_v'].append(s_.center[1])
                #
                shapes['flagout'].append(flag)

            shapes['flagout'] = np.asarray(shapes['flagout'], dtype="bool")
            shapes['sigma']   = np.asarray(shapes['sigma'], dtype="float")
            shapes['sigma_normalized'] = shapes['sigma']/ getattr(np,normalisation)(shapes['sigma'][~shapes['flagout']])
            
            self.shapes[which] = pandas.DataFrame(shapes)
            
    # --------- #
    #  GETTER   #
    # --------- #
    def get_residual(self):
        """ """
        return self._starsarray - self._psfmodelarray

    # --------- #
    #  PLOTTER  #
    # --------- #
    def show_psfmodeling(self, index=None, axes=None, title="", tight_layout=True, **kwargs):
        """ """
        if index is None:
            index = np.random.randint(0, self.nstars)
            
        return show_psfmodeling(self.stars[index], self.psfmodel[index],
                                    axes=axes, title=title, tight_layout=tight_layout,
                                    **kwargs)
    
    def show_shape(self, which="T", vmin="2", vmax="98", cmap='RdBu_r', normres=True,
                       tight_layout=True, rmflagout=True, cvmin=None, cvmax=None):
        """ """
        import matplotlib.pyplot as mpl
        from .utils import vminvmax_parser
        #
        # - Input
        if cvmax is None:
            cvmax = vmax
        if cvmin is None:
            cvmin = vmin
        # - end: Input
        #

        fig = mpl.figure(figsize=[9,3])
        left,width = 0.02, 0.25
        hspan, hxspan = 0.015,0.07

        axd = fig.add_axes([left+0*(width+hspan)     ,0.1,width,0.8])
        axm = fig.add_axes([left+1*(width+hspan)     ,0.1,width,0.8])       
        axc = fig.add_axes([left+2*(width+hspan)     ,0.1,0.01,0.8])
        axr = fig.add_axes([left+2*(width+hspan) +hxspan,0.1,width,0.8])        
        axcr = fig.add_axes([left+3*(width+hspan)+hxspan,0.1,0.01,0.8])
        axes = [axd,axm,axr]

        # Data to show
        data  = np.asarray(self.shapes["stars"][which])
        model = np.asarray(self.shapes["psfmodel"][which]) if not rmflagout else np.asarray(self.shapes["psfmodel"][which])[~self.shapes["psfmodel"]["flagout"]]
        u, v  = np.asarray(self.u), np.asarray(self.v)
        if rmflagout:
            print("removing the outliers")
            flagout = self.shapes["stars"]["flagout"] | self.shapes["psfmodel"]["flagout"]
            print(f"from {len(data)} targets")
            data = data[~flagout]
            model = model[~flagout]
            u, v = u[~flagout], v[~flagout]
            print(f"to {len(data)} targets")

        vmin_, vmax_ = vminvmax_parser(data, vmin, vmax)

        # Properties        
        scat_kwargs = {'cmap':cmap, 's':30}
        proptext = dict(fontsize="medium", color="0.3", loc="left")
        
        s = axd.scatter(u, v, c=data, vmin=vmin_, vmax=vmax_, **scat_kwargs)
        axd.set_title('stars', **proptext)

        s = axm.scatter(u, v, c=model,vmin=vmin_, vmax=vmax_, **scat_kwargs)
        fig.colorbar(s,cax=axc)
        axm.set_title('model', **proptext)

        #
        residual = data-model
        if normres:
            residual /= model
            
        vminr, vmaxr = vminvmax_parser(residual, cvmin, cvmax)
        s = axr.scatter(u, v, c=residual, vmin=vminr, vmax=vmaxr, **scat_kwargs)
        fig.colorbar(s,cax=axcr)
        
        axes[2].set_title('(data-model)/model', **proptext)
        
        [[ax_.set_xticks([]),ax_.set_yticks([])] for ax_ in axes]
    
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def stars(self):
        """ """
        return self._stars

    def has_stars(self):
        """ """
        return hasattr(self, "_stars") and self.stars is not None
    
    @property
    def nstars(self):
        """ """
        return len(self.stars)
    
    @property
    def u(self):
        """ """
        return self._u
    
    @property
    def v(self):
        """ """
        return self._v
    
    @property
    def psfmodel(self):
        """ """
        return self._psfmodel

    @property
    def shapes(self):
        """ """
        if not hasattr(self,"_shapes"):
            self._shapes = {}
        return self._shapes
