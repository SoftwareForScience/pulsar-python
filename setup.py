# -*- coding: utf-8 -*-

"""Setup configuration file. Used to describe the Asteria Package.  """

from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

with open('LICENSE') as f:
    LICENSE = f.read()

setup(
    name='asteria',
    version='0.0.1',
    description='An open-source DSP library for pulsar detection.',

    long_description=README,
    author='Amsterdam University of Applied Sciences',
    author_email='',
    url='https://github.com/AUAS-Pulsar/Asteria',
    license=LICENSE,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)
