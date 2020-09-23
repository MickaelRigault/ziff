#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          setupy.py
# Description:       script description
# Author:            Romain Graziani <romain.graziani@clermont.in2p3.fr>
# Author:            $Author: rgraziani $
# Created on:        $Date: 2020/09/21 11:04:55 $
# Modified on:       2020/09/21 11:10:37
# Copyright:         2019, Romain Graziani
# $Id: setupy.py, 2020/09/21 11:04:55  RG $
################################################################################

"""
.. _setupy.py:

setupy.py
==============


"""
__license__ = "2019, Romain Graziani"
__docformat__ = 'reStructuredText'
__author__ = 'Romain Graziani <romain.graziani@clermont.in2p3.fr>'
__date__ = '2020/09/21 11:04:55'
__adv__ = 'setupy.py'

from distutils.core import setup

setup(name='Ziff',
      version='1.0',
      description='Piff for ZTF',
      author='Romain Graziani',
      author_email='romain.graziani@clermont.in2p3.fr',
      url='https://github.com/rgraz/Ziff',
      packages=['ziff'],
      package_data={'ziff': ['data/*']}
     )
# End of setupy.py ========================================================
