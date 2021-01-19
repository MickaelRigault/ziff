#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          plots.py
# Description:       script description
# Author:            Romain Graziani <romain.graziani@clermont.in2p3.fr>
# Author:            $Author: rgraziani $
# Created on:        $Date: 2020/10/02 10:46:12 $
# Modified on:       2020/10/02 14:41:31
# Copyright:         2019, Romain Graziani
# $Id: plots.py, 2020/10/02 10:46:12  RG $
################################################################################

"""
.. _plots.py:

plots.py
==============


"""
__license__ = "2019, Romain Graziani"
__docformat__ = 'reStructuredText'
__author__ = 'Romain Graziani <romain.graziani@clermont.in2p3.fr>'
__date__ = '2020/10/02 10:46:12'
__adv__ = 'plots.py'

import matplotlib.pyplot as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic_2d
import numpy as np

def make_focal_plane():
    fig = mpl.figure(figsize=(8,7.75))
    gs = GridSpec(4, 5, figure=fig,wspace=0.1,hspace=0.1, width_ratios = [1,1,1,1,0.1], height_ratios=[1,1,1,1])
    return fig, gs

def get_ax_cbar(fig, gs):
    ax = fig.add_subplot(gs[:,-1])
    return ax

def get_ax_ccd(fig, gs, ccd):
    j = ((ccd-1) % 4)
    i = 3- ((ccd-1) // 4)
    ax = fig.add_subplot(gs[i,j])
    return ax, i, j


def vminvmax_parser(data, vmin, vmax):
    """ """
    if vmin is None:
        vmin="0"
    if vmax is None:
        vmax = "100"
    if type(vmin) == str:
        vmin = np.percentile(data, float(vmin))
    if type(vmax) == str:
        vmax = np.percentile(data, float(vmax))
    return vmin, vmax

def get_threeplot_axes(fig=None, left= 0.08, width=0.22, hspan=0.01,
                        extra_hspan=0.1, bottom=0.2,
                        height = 0.7, cwidth = 0.01):
    """ """
    if fig is None:
        fig = mpl.figure(figsize=[10,3])
        
    left, width, hspan = 0.08, 0.22, 0.01
    extra_hspan = 0.1
    bottom, height = 0.2,0.7
    cwidth = 0.01
    
    axd = fig.add_axes([left+0*(width+ hspan), bottom,  
                        width, height])
    axm = fig.add_axes([left+1*(width+ hspan), bottom,  
                        width, height])
    caxm = fig.add_axes([left+2*(width+ hspan), bottom,  
                        cwidth, height])
    axr = fig.add_axes([left+2*(width+ hspan)+extra_hspan+hspan, bottom,  
                        width, height])
    caxr = fig.add_axes([left+3*(width+ hspan)+extra_hspan+hspan, bottom,  
                        cwidth, height])

    return [axd, axm, caxm], [axr, caxr]


def show_shapebinned(dataframe, nbins=50, 
                     which="sigma", vmin="2", vmax="98", cmap='RdBu_r', normres=True,
                     tight_layout=True, rmflagout=True, cvmin=None, cvmax=None, suffix=["_stars","_model"],
                     normed_by=["ccdid", "obsjd"], ref="_model",normstat="median",
                     rvmin=-0.05, rvmax=0.05,
              ):
    """ """
    import matplotlib.pyplot as mpl
    from ziff.utils import vminvmax_parser
    from scipy.stats import binned_statistic_2d
    #
    # - Input
    if cvmax is None:
        cvmax = vmax
    if cvmin is None:
        cvmin = vmin
    # - end: Input
    #
    
    #
    if rmflagout:
        dataframe = dataframe[~(dataframe[f"flagout{suffix[0]}"] | dataframe[f"flagout{suffix[0]}"])]
        
    u,v, data, model = dataframe[["u_model","v_stars",f"{which}{suffix[0]}",f"{which}{suffix[1]}"]].values.T
    residual = (data-model)/model
    
    if normed_by is not None:
        grouped = dataframe.groupby(normed_by)
        normalisation = grouped[f"{which}{ref}"].transform(normstat)
        data  = data/normalisation
        model = model/normalisation
    

    #
    
    fig = mpl.figure(figsize=[10,3])
    left,width = 0.05, 0.25
    hspan, hxspan = 0.01,0.08

    axd = fig.add_axes([left+0*(width+hspan)     ,0.1,width,0.8])
    axm = fig.add_axes([left+1*(width+hspan)     ,0.1,width,0.8])       
    axc = fig.add_axes([left+2*(width+hspan)     ,0.1,0.01,0.8])
    axr = fig.add_axes([left+2*(width+hspan) +hxspan,0.1,width,0.8])        
    axcr = fig.add_axes([left+3*(width+hspan)+hxspan,0.1,0.01,0.8])
    axes = [axd,axm,axr]

    # Data to show

    vmin_, vmax_ = vminvmax_parser(model, vmin, vmax)

    
    bins_u = np.linspace(np.min(u),np.max(u),nbins)
    bins_v =  np.linspace(np.min(v),np.max(v),nbins)
    
    data_bstat = binned_statistic_2d(u, v, data, statistic="median", bins=[bins_u,bins_v]).statistic
    model_bstat = binned_statistic_2d(u, v, model, statistic="median", bins=[bins_u,bins_v]).statistic
    res_bstat = binned_statistic_2d(u, v, residual, statistic="median", bins=[bins_u,bins_v]).statistic
    
    
    default_imshow_kwargs =  {'cmap' : cmap, 'origin':'lower', 
                              'extent' : (bins_u[0],bins_u[-1],bins_v[0],bins_v[-1]),
                              "vmin":vmin_, "vmax":vmax_}
    
    im = axd.imshow(data_bstat.T,  **default_imshow_kwargs)
    im = axm.imshow(model_bstat.T, **default_imshow_kwargs)
    fig.colorbar(im,cax=axc)
    im = axr.imshow(res_bstat.T, **{**default_imshow_kwargs,**{"vmin":rvmin, "vmax":rvmax}})
    fig.colorbar(im,cax=axcr)
    
    # Main axes
    [ax_.tick_params(labelsize="small", labelcolor="0.5") for ax_ in [axm, axd, axr]]
    # Color bars
    [ax_.tick_params(labelsize="medium", labelcolor="k") for ax_ in [axc, axcr]]
    axm.set_yticks([])
    axr.set_yticks([])
#    [[ax_.set_xticks([]),ax_.set_yticks([])] for ax_ in axes]
    
# End of plots.py ========================================================
