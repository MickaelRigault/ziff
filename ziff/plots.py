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

def get_threeplot_axes(fig=None, scalex=1, scaley=1, globalbottom=0, globalleft=0,
                        left= 0.05, width=0.25, bottom=0.15, heigth=0.75,
                        hspan=0.015, hxspan=0.1,
                        cwidth = 0.01):
    """ """
    if fig is None:
        fig = mpl.figure(figsize=[9,3])
        
    left, width = globalleft+left*scalex, width*scalex
    bottom, heigth = globalbottom+bottom*scaley, heigth*scaley
    hspan, hxspan = hspan*scalex,hxspan*scalex
    cwidth = cwidth*scalex

    axd = fig.add_axes([left+0*(width+hspan)     ,bottom,width,heigth])
    axm = fig.add_axes([left+1*(width+hspan)     ,bottom,width,heigth])       
    cax = fig.add_axes([left+2*(width+hspan)     ,bottom,cwidth,heigth])
    axr = fig.add_axes([left+2*(width+hspan) +hxspan,bottom,width,heigth])        
    caxr = fig.add_axes([left+3*(width+hspan)+hxspan,bottom,cwidth,heigth])
    axes = [axd,axm,axr]
    caxes = [cax, caxr]
    
    return fig, axes, caxes


def display_binned2d(ax, binneddata, xbins=None, ybins=None, vmin="5", vmax="95", cax=None,
                         transpose=False, cmap=None, **kwargs):
    """ """
    from ziff.plots import vminvmax_parser
    vmin_, vmax_ = vminvmax_parser(binneddata, vmin, vmax)
    if xbins is not None and ybins is not None:
        extent = (xbins[0],xbins[-1],ybins[0],ybins[-1])
    else:
        extent = None
        
    prop = dict(origin="lower", extent=extent)
    im = ax.imshow(binneddata.T if transpose else binneddata, cmap=cmap,
                    vmin=vmin_, vmax=vmax_, **{**prop,**kwargs})
    
    if cax is not None:
        ax.figure.colorbar(im, cax=cax)
        
    return im

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



#
#   Plotting
#
def show_psfresults(psf, nstars=9, index=None, savefile=None, **kwargs):
    """ 
    if index given, nstars is ignored. 
    """
    all_indexes = np.arange(len(psf.stars))
    if index is not None:
        nstars = len(index)
    elif nstars is not None:
        index = np.random.choice(all_indexes, nstars, replace=False)
    else:
        raise ValueError("you must provide easer nstars or index ; both are None")
        
    print(index)
    
    fig = mpl.figure(figsize=[9,3*nstars])
    for i, index_ in enumerate(index):
        _ = show_single_psfstar(psf, index_, fig=fig, 
                            globalbottom=i/nstars, scaley=1/nstars, **kwargs)
        if i ==0:
            fig, axes, cax = _
    
    axes[0].set_xlabel("Data", weight="medium")
    axes[1].set_xlabel("Model", weight="medium")
    axes[2].set_xlabel("Residual", weight="medium")
    
    if savefile:
        fig.savefig(savefile)
        
    return fig

def show_single_psfstar(psf, index, fig=None,
                        scalex=1, scaley=1, globalbottom=0, globalleft=0,
                       vmin="1", vmax="99",cvmin="1", cvmax="99",
                        show_text=True, **kwargs):
    """ """
    fig, axes, caxes= get_threeplot_axes(fig=fig, scalex=scalex, scaley=scaley,
                                             globalbottom=globalbottom, globalleft=globalleft)
    [axd,axm,axr] = axes
    [cax, caxr] = caxes

    # Star and Model
    star_plotted  = psf.stars[index]
    model_plotted = psf.drawStar(star_plotted)


    data = star_plotted.image.array
    model = model_plotted.image.array
    residual = data-model

    vmin_, vmax_ = vminvmax_parser(data, vmin, vmax)
    cvmin_, cvmax_ = vminvmax_parser(residual, cvmin, cvmax)

    prop = dict(origin="lower", cmap="cividis", vmin=vmin_, vmax=vmax_)

    # Data
    scd = axd.imshow(data, **{**prop,**kwargs})
    fig.colorbar(scd, cax=cax)
    # Model
    scm = axm.imshow(model, **prop)
    fig.colorbar(scm, cax=cax)
    # Dif
    scr = axr.imshow(residual, **{**prop,**{"vmin":cvmin_, "vmax":cvmax_}, **kwargs})
    fig.colorbar(scr, cax=caxr)

    [[ax_.set_xticks([]),ax_.set_yticks([])] for ax_ in axes]
    [ax_.tick_params(labelsize="small") for ax_ in caxes]
    # - Text
    if show_text:
        axd.text(-0.02, 0.5, f"index: {index} (x={star_plotted.image.center.x}, y={star_plotted.image.center.y})",
                 transform=axd.transAxes, rotation=90, va="center", ha="right", fontsize="x-small", color="0.3")
        
    return fig, axes, caxes


# End of plots.py ========================================================



    
