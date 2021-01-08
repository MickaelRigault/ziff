#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. _setupy.py:

setupy.py
==============


"""
from distutils.core import setup

setup(name='Ziff',
      version='0.2.1',
      description='Piff for ZTF',
      author='Mickael Rigault, Romain Graziani',
      author_email='m.rigault@ipnl.in2p3.fr',
      url='https://github.com/MickaelRigault/Ziff',
      packages=['ziff'],
      package_data={'ziff': ['data/*']},
      scripts=["bin/ziffit.py"]
    #['ziff/scripts/run_ccd.py','ziff/scripts/download_query.py','ziff/scripts/download_target.py']
     )
# End of setupy.py ========================================================
