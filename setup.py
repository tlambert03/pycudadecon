# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from codecs import open
from os import path


HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()


setup(
    name="pycudadecon",
    version="0.0.11",
    description="Python wrapper for CUDA-accelerated 3D deconvolution",
    long_description=README,
    author="Talley Lambert",
    author_email="talley.lambert@gmail.com",
    url="https://github.com/tlambert03/pycudadecon",
    license="MIT",
    packages=find_packages(exclude=("tests")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=["numpy", "tifffile"],
)
