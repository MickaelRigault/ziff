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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic_2d


def make_focal_plane():
    fig = plt.figure(figsize=(8,7.75))
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
    
# End of plots.py ========================================================
