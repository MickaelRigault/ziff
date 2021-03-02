
"""
.. module:: pixelmodelconvolved
"""

from __future__ import print_function
import numpy as np
import galsim

from scipy.ndimage import gaussian_filter
from piff.pixelgrid import PixelGrid
from piff.star import Star, StarData, StarFit


class ConvolvedPixelGrid( PixelGrid ):
    """ """
    _EXTRA_TERM = 1

    def __init__(self, scale, size, interp=None, centered=True, logger=None,
                 start_sigma=None, degenerate=None, **kwargs):
        """ """
        _ = super().__init__(scale, size, interp=interp, centered=centered,
                                 logger=logger, start_sigma=start_sigma,
                                 degenerate=degenerate, **kwargs)
        
        logger = galsim.config.LoggerWrapper(logger)
        self._nparams = size*size + self._EXTRA_TERM
        logger.debug("change nparams = %d (convoltion)",self._nparams)
        self._nparams_pixelgrid = size*size
        logger.debug("change nparams pixelgrid = %d (convoltion)", self._nparams_pixelgrid)        

    def _indexFromPsfxy(self, psfx, psfy):
        """ Turn arrays of coordinates of the PSF array into a single same-shape
        array of indices into a 1d parameter vector.  The index is <0 wherever
        the psf x,y values were outside the PSF mask.

        :param psfx:  array of integer x displacements from origin of the PSF grid
        :param psfy:  array of integer y displacements from origin of the PSF grid

        :returns: same shape array, filled with indices into 1d array
        """
        # Shift psfy, psfx to reference a 0-indexed array
        y = psfy + self._origin[0]
        x = psfx + self._origin[1]

        # Good pixels are where there is a valid index
        # Others are set to -1.
        ind = np.ones_like(psfx, dtype=int) * -1
        good = (0 <= y) & (y < self.size) & (0 <= x) & (x < self.size)
        indices = np.arange(self._nparams_pixelgrid, dtype=int).reshape(self.size,self.size)
        ind[good] = indices[y[good], x[good]]
        return ind
        
    def initialize(self, star, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a star instance with the appropriate initial fit values
        """
        data, weight, u, v = star.data.getDataVector()
        # Start with the sum of pixels as initial estimate of flux.
        flux = np.sum(data)
        if self._centered:
            # Initial center is the centroid of the data.
            Ix = np.sum(data * u) / flux
            Iy = np.sum(data * v) / flux
            center = (Ix,Iy)
        else:
            # In this case, center is fixed.
            center = star.fit.center

        # Calculate the second moment to initialize an initial Gaussian profile.
        # hsm returns: flux, x, y, sigma, g1, g2, flag
        sigma = star.hsm[3]

        # Create an initial parameter array using a Gaussian profile.
        u = np.arange( -self._origin[0], self.size-self._origin[0]) * self.scale
        v = np.arange( -self._origin[1], self.size-self._origin[1]) * self.scale
        rsq = (u*u)[:,np.newaxis] + (v*v)[np.newaxis,:]
        gauss = np.exp(-rsq / (2.* sigma**2))
        # change params -> params_pixelgrid
        params_pixelgrid = gauss.ravel() 

        # Normalize to get unity flux
        params_pixelgrid /= np.sum(params_pixelgrid)*self.pixel_area
        
        # + CHANGE: add an extra term ; sigma convolve 0 by default
        params = np.append(params_pixelgrid, np.ones(self._EXTRA_TERM)*1e-5)

        starfit = StarFit(params, flux, center)
        return Star(star.data, starfit)


    # def fit(self, star, logger=None): # no change
    
    def chisq(self, star, logger=None):
        """Calculate dependence of chi^2 = -2 log L(D|p) on PSF parameters for single star.
        as a quadratic form chi^2 = dp^T AT A dp - 2 bT A dp + chisq,
        where dp is the *shift* from current parameter values.  Returned Star
        instance has the resultant A, b, chisq, flux, center) attributes,
        but params vector has not have been updated yet (could be degenerate).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated StarFit
        """
        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()

        # Subtract star.fit.center from u, v:
        u -= star.fit.center[0]
        v -= star.fit.center[1]

        # Only use pixels covered by the model.
        mask = (np.abs(u) <= self.maxuv) & (np.abs(v) <= self.maxuv)
        data = data[mask]
        weight = weight[mask]
        u = u[mask]
        v = v[mask]

        # Compute the full set of coefficients for each pixel in the data
        # The returned arrays here are Ndata x Ninterp.
        # Each column column corresponds to a different x,y value in the model that could
        # contribute information about the given data pixel.
        coeffs, psfx, psfy = self.interp_calculate(u/self.scale, v/self.scale)
        # -> coeffs = size**2
        
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        alt_index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        # This just makes it easier to do the sums, since then the nopsf values won't contribute.
        coeffs = np.where(nopsf, 0., coeffs)

        # Multiply kernel (and derivs) by current PSF element values to get current estimates
        pvals = self.get_pixelparams(star.fit.params, flatten=True)[alt_index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        scaled_flux = star.fit.flux * star.data.pixel_area
        resid = data - mod * scaled_flux

        # Now construct A, b, chisq that give chisq vs linearized model.
        #
        # We can say we are looking for a weighted least squares solution to the problem
        #
        #   A dp = b
        #
        # where b is the array of residuals, and A_ij = coeffs[i][k] iff alt_index1d[i][k] == j.
        #
        # The weights are dealt with in the standard way, by multiplying both A and b by sqrt(w).

        A = np.zeros((len(data), self._nparams), dtype=float)
        for i in range(len(data)):
            ii = index1d[i,:]
            cc = coeffs[i,:]
            # Select only those with ii >= 0
            cc = cc[ii>=0] * scaled_flux
            ii = ii[ii>=0]
            A[i,ii] = cc
        sw = np.sqrt(weight)
        Aw = A * sw[:,np.newaxis]
        bw = resid * sw
        chisq = np.sum(bw**2)
        dof = np.count_nonzero(weight)

        outfit = StarFit(star.fit.params,
                         flux = star.fit.flux,
                         center = star.fit.center,
                         params_var = star.fit.params_var,
                         chisq = chisq,
                         dof = dof,
                         A = Aw,
                         b = bw)

        return Star(star.data, outfit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      The fit parameters for a given star.

        :returns: a galsim.GSObject instance
        """
        im = galsim.Image(self.get_pixelparams(params, flatten=False), scale=self.scale)
        return galsim.InterpolatedImage(im, x_interpolant=self.interp,
                                        normalization='sb', use_true_center=False, flux=1.)

    def normalize(self, star):
        """Make sure star.fit.params are normalized properly.

        Note: This modifies the input star in place.
        """
        # Backwards compatibility check.
        # We used to only keep nparams - 1 or nparams - 3 values in fit.params.
        # If this is the case, fix it up to match up with our new convention.
        nparams1 = len(star.fit.params)
        nparams2 = self.size**2
        if nparams1 != nparams2+self._EXTRA_TERM:
            raise NotImplementedError(f"size of parameter {nparams1} don't match the expectation {nparams2}+{self._EXTRA_TERM}")

        # Normally this is all that is required.
        star.fit.params[:self._EXTRA_TERM] /= np.sum(star.fit.params[:self._EXTRA_TERM]
                                                    )*self.pixel_area

    def reflux(self, star, fit_center=True, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  DOF in the result assume only flux (& center) are free parameters.

        :param star:        A Star instance
        :param fit_center:  If False, disable any motion of center
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance, with updated flux, center, chisq, dof
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Reflux for star:")
        logger.debug("    flux = %s",star.fit.flux)
        logger.debug("    center = %s",star.fit.center)
        logger.debug("    props = %s",star.data.properties)
        logger.debug("    image = %s",star.data.image)
        #logger.debug("    image = %s",star.data.image.array)
        #logger.debug("    weight = %s",star.data.weight.array)
        logger.debug("    image center = %s",star.data.image(star.data.image.center))
        logger.debug("    weight center = %s",star.data.weight(star.data.weight.center))

        # Make sure input is properly normalized
        self.normalize(star)
        scaled_flux = star.fit.flux * star.data.pixel_area
        center = star.fit.center

        # Calculate the current centroid of the model at the location of this star.
        # We'll shift the star's position to try to zero this out.
        delta_u = np.arange(-self._origin[0], self.size-self._origin[0])
        delta_v = np.arange(-self._origin[1], self.size-self._origin[1])
        u, v = np.meshgrid(delta_u, delta_v)
        temp = self.get_pixelparams(star.fit.params, flatten=False)
        params_cenu = np.sum(u*temp)/np.sum(temp)
        params_cenv = np.sum(v*temp)/np.sum(temp)

        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()

        u -= center[0]
        v -= center[1]
        mask = (np.abs(u) <= self.maxuv) & (np.abs(v) <= self.maxuv)
        data = data[mask]
        weight = weight[mask]
        u = u[mask]
        v = v[mask]

        # Build the model and maybe also d(model)/dcenter
        # This tracks the same steps in chisq.
        # TODO: Make a helper function to consolidate the common code.
        if self._centered:
            coeffs, psfx, psfy, dcdu, dcdv = self.interp_calculate(u/self.scale, v/self.scale,
                                                                  True)
            dcdu /= self.scale
            dcdv /= self.scale
        else:
            coeffs, psfx, psfy = self.interp_calculate(u/self.scale, v/self.scale)
        
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        alt_index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)
        if self._centered:
            dcdu = np.where(nopsf, 0., dcdu)
            dcdv = np.where(nopsf, 0., dcdv)

        # Multiply kernel (and derivs) by current PSF element values to get current estimates
        pvals = self.get_pixelparams(star.fit.params, flatten=True)[alt_index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        if self._centered:
            dmdu = scaled_flux * np.sum(dcdu*pvals, axis=1)
            dmdv = scaled_flux * np.sum(dcdv*pvals, axis=1)
            derivs = np.vstack((mod, dmdu, dmdv)).T
        else:
            # In this case, we're just making it a column vector.
            derivs = np.vstack((mod,)).T
        resid = data - mod*scaled_flux
        logger.debug("total pixels = %s, nopsf = %s",len(pvals),np.sum(nopsf))

        # Now construct the design matrix for this minimization
        #
        #    A x = b
        #
        # where x = [ dflux, duc, dvc ]^T or just [ dflux ] and b = resid.
        #
        # A[0] = d( mod * flux ) / dflux = mod
        # A[1] = d( mod * flux ) / duc   = flux * sum(dcdu * pvals, axis=1)
        # A[2] = d( mod * flux ) / dvc   = flux * sum(dcdv * pvals, axis=1)

        # For large matrices, it is generally better to solve this with QRP, but with this
        # small a matrix, it is faster and not any less stable to just compute AT A and AT b
        # and solve the equation
        #
        #    AT A x = AT b

        Atw = derivs.T * weight  # weighted least squares
        AtA = Atw.dot(derivs)
        Atb = Atw.dot(resid)
        x = np.linalg.solve(AtA, Atb)
        chisq = np.sum(resid**2 * weight)
        dchi = Atb.dot(x)
        logger.debug("chisq = %s - %s => %s",chisq,dchi,chisq-dchi)

        # update the flux (and center) of the star
        logger.debug("initial flux = %s",scaled_flux)
        scaled_flux += x[0]
        logger.debug("flux += %s => %s",x[0],scaled_flux)
        logger.debug("center = %s",center)
        if self._centered:
            center = (center[0]+x[1], center[1]+x[2])
            logger.debug("center += (%s,%s) => %s",x[1],x[2],center)

            # In addition to shifting to the best fit center location, also shift
            # by the centroid of the model itself, so the next next pass through the
            # fit will be closer to centered.  In practice, this converges pretty quickly.
            center = (center[0]+params_cenu*self.scale, center[1]+params_cenv*self.scale)
            logger.debug("params cen = %s,%s.  center => %s",params_cenu,params_cenv,center)

        dof = np.count_nonzero(weight)
        logger.debug("dchi, dof, do_center = %s, %s, %s", dchi, dof, self._centered)

        # Update to the expected new chisq value.
        chisq = chisq - dchi
        return Star(star.data, StarFit(star.fit.params,
                                       flux = scaled_flux / star.data.pixel_area,
                                       center = center,
                                       params_var = star.fit.params_var,
                                       chisq = chisq,
                                       dof = dof,
                                       A = star.fit.A,
                                       b = star.fit.b))

    # def interp_calculate(self, u, v, derivs=False): # No change
    # def _kernel1d(self, u): # No change
        
        
    # --------------- #
    #  Extra methods  #
    # --------------- #
    def get_pixelparams(self, params, flatten=True):
        """ """
        im_params, extra_params = params[:-self._EXTRA_TERM], params[-self._EXTRA_TERM:]
        sigma_convol = np.abs(extra_params[0])
        
        params_sq  = gaussian_filter(im_params.reshape(self.size,self.size), sigma_convol)
        if flatten:
(            return params_sq.reshape(self.size*self.size)
                 
        return params_sq
