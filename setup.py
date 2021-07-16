#! /usr/bin/env python3
import os

from setuptools import setup, find_namespace_packages

PROJECT_DIR = os.path.dirname(__file__)

INFO = open(os.path.join(PROJECT_DIR, 'INFO')).readlines()
INFO = dict((l.strip().split('=') for l in INFO))

DEPENDENCIES = open(os.path.join(PROJECT_DIR, 'requirements.txt')).readlines()

setup(name='myra-core',
      version=INFO['version'],
      author=INFO['author'],
      author_email=INFO['author_email'],
      url=INFO['url'],
      python_requires='>=3',
      entry_points={
          'console_scripts': ['core=ink.core:main']
      },      
      packages=find_namespace_packages(include=['ink.core.*']),
      install_requires=[d for d in DEPENDENCIES if '://' not in d],
      package_data={'ink.core': ['templates/*']},
      zip_safe=False)
