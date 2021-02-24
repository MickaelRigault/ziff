#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. _setupy.py:

setupy.py
==============


"""
from distutils.core import setup
from setuptools import setup, find_packages


packages = find_packages()
print(f"packages to be installed: {packages}")


VERSION = '0.3.2'
        
setup(name='ziff',
      version=VERSION,
      description='Piff for ZTF',
      author='Mickael Rigault, Romain Graziani',
      author_email='m.rigault@ipnl.in2p3.fr',
      url='https://github.com/MickaelRigault/Ziff',
      packages=packages,
      package_data={'ziff': ['data/*']},
      scripts=["bin/ziffit.py","bin/qsub_ziffit.sh"]
    #['ziff/scripts/run_ccd.py','ziff/scripts/download_query.py','ziff/scripts/download_target.py']
     )
# End of setupy.py ========================================================


