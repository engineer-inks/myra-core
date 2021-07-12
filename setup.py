#! /usr/bin/env python3
import os

from setuptools import setup

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
      packages=['myra.dna'],
      namespace_packages=['myra', 'myra.dna'],
      install_requires=[d for d in DEPENDENCIES if '://' not in d],
      zip_safe=False)
