#!/usr/bin/env python

from setuptools import setup

setup(name='gwmemory',
      version='0.1.3',
      packages=['gwmemory'],
      package_dir={'gwmemory': 'gwmemory'},
      package_data={'gwmemory': ['data/gamma_coefficients*.dat', 'data/*WEB.dat']},
      install_requires=['numpy', 'scipy', 'pandas', 'glob2', 'deepdish', 'lalsuite']
      )
