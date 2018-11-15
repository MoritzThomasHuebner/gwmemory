#!/usr/bin/env python

from setuptools import setup

setup(name='gwmemoritz',
      version='0.1.0',
      packages=['gwmemory'],
      package_dir={'gwmemory': 'gwmemory'},
      package_data={'gwmemory': ['data/gamma_coefficients*.dat', 'data/*WEB.dat']},
      )
