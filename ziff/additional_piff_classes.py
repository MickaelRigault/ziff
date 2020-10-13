#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          additional_piff_classes.py
# Description:       script description
# Author:            Romain Graziani <romain.graziani@clermont.in2p3.fr>
# Author:            $Author: rgraziani $
# Created on:        $Date: 2020/10/13 10:42:21 $
# Modified on:       2020/10/13 10:43:38
# Copyright:         2019, Romain Graziani
# $Id: additional_piff_classes.py, 2020/10/13 10:42:21  RG $
################################################################################

"""
.. _additional_piff_classes.py:

additional_piff_classes.py
==============


"""
__license__ = "2019, Romain Graziani"
__docformat__ = 'reStructuredText'
__author__ = 'Romain Graziani <romain.graziani@clermont.in2p3.fr>'
__date__ = '2020/10/13 10:42:21'
__adv__ = 'additional_piff_classes.py'

from piff.basis_interp import BasisPolynomial
import numpy as np
import pickle
#from scipy.interpolate import LinearNDInterpolator


class BasisPolynomialPlusMap(BasisPolynomial):
    def __init__(self, order, interpolation_map_file, keys=('u','v'), max_order=None, use_qr=False, logger=None):
        super(BasisPolynomialPlusMap, self).__init__(order, keys=('u','v'), max_order=None, use_qr=False, logger=None)
        # Now build a mask that picks the desired polynomial products
        # Start with 1d arrays giving orders in all dimensions
        ord_ranges = [np.arange(order+1,dtype=int) for order in self._orders]
        # Nifty trick to produce n-dim array holding total order
        sumorder = np.sum(np.ix_(*ord_ranges))
        self._mask = sumorder <= self._max_order + 1 # +1 is for the map
        self._interpolation_map_file = interpolation_map_file
        self.kwargs = {
            'order' : order,
            'use_qr' : use_qr,
            'interpolation_map_file' : interpolation_map_file
        }
        self.load_map()
         

    def load_map(self):
        with open(self.kwargs['interpolation_map_file'], 'rb') as f:
            interp = pickle.load(f)
        self._map = interp
        
    def basis(self, star):
        """Return 1d array of polynomial basis values for this star

        :param star:   A Star instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        polynomial_basis = super().basis(star)
        # We add one coefficient to the polynomial basis coming from the map
        map_value = self._map(self.getProperties(star))
        return np.hstack([polynomial_basis, map_value])
    

    def constant(self, value=1.):
        """Return 1d array of coefficients that represent a polynomial with constant value.

        :param value:  The value to use as the constant term.  [default: 1.]

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        out = np.zeros( np.count_nonzero(self._mask) + 1, dtype=float)
        out[0] = value  # The constant term is always first.
        return out

# End of additional_piff_classes.py ========================================================
