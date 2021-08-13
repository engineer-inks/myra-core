#! /usr/bin/env python3
import os

from setuptools import setup

PROJECT_DIR = os.path.dirname(__file__)

INFO = open(os.path.join(PROJECT_DIR, 'INFO')).readlines()
INFO = dict((l.strip().split('=') for l in INFO))

DEPENDENCIES = open(os.path.join(PROJECT_DIR, 'requirements.txt')).readlines()

setup(name='ink-core-forge',
      version=INFO['version'],
      author=INFO['author'],
      author_email=INFO['author_email'],
      url=INFO['url'],
      python_requires='>=3.8',
      entry_points={
          'console_scripts': ['core=ink.core.forge:main']
      },      
      packages=['ink.core.forge'],
      namespace_packages=['ink', 'ink.core'],
      install_requires=[d for d in DEPENDENCIES if '://' not in d],
      package_data={'ink.core.forge': ['joins/*']},
      zip_safe=False)
