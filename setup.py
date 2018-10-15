# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='asteria',
    version='0.0.1',
    description='An open-source DSP library for pulsar detection.',
    long_description=readme,
    author='Amsterdam University of Applied Sciences',
    author_email='',
    url='https://github.com/AUAS-Pulsar/Asteria',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)
