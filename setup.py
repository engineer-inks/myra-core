#! /usr/bin/env python3
import os

from setuptools import setup

PROJECT_DIR = os.path.dirname(__file__)

INFO = open(os.path.join(PROJECT_DIR, 'INFO')).readlines()
INFO = dict((l.strip().split('=') for l in INFO))

DEPENDENCIES = open(os.path.join(PROJECT_DIR, 'requirements.txt')).readlines()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('ink/core/forge/joins')

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
      package_data={'ink.core.forge': ['templates/*'],'ink.core.forge.*:': extra_files},
      zip_safe=False)
