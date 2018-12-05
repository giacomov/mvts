#!/usr/bin/env python

from setuptools import setup

# Get the version number
with open('mvts/version.py') as f:
    ext_code = compile(f.read(), "mvts/version.py", 'exec')
    exec(ext_code)

# Normal packages

packages = ['mvts']

setup(
    name="mvts",

    packages=packages,

    # data_files=[('astromodels/data/functions', glob.glob('astromodels/data/functions/*.yaml'))],

    # The __version__ comes from the exec at the top

    version=__version__,

    description="Make wavelet spectrum to find the Minimum Variability Time Scale in a light curve",

    author='Giacomo Vianello',

    author_email='giacomo.vianello@gmail.com',

    url='https://github.com/giacomov/mvts',

    download_url='https://github.com/giacomov/mvts/archive/v%s' % __version__,

    keywords=['Transients', 'Time-domain astronomy', 'wavelets','minimum variability'],

    classifiers=[],

    install_requires=[
        'pycwt',
        'numexpr',
        'numpy',
        'astropy',
        'matplotlib'],

    ext_modules=[],
)
