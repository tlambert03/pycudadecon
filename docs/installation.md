# Installation

The conda package includes the required pre-compiled libraries for Windows and
Linux.

*macOS is not supported*

```bash
conda install -c conda-forge pycudadecon
```

## GPU requirements

This software requires a CUDA-compatible NVIDIA GPU. The underlying cudadecon
libraries have been compiled against different versions of the CUDA toolkit.
The required CUDA libraries are bundled in the conda distributions so you don't
need to install the CUDA toolkit separately.  If desired, you can pick which
version of CUDA you'd like based on your needs, but please note that different
versions of the CUDA toolkit have different GPU driver requirements:

To specify a specific cudatoolkit version, install as follows:

```sh
# install with cudatoolkit=10.2
conda install -c condaforge pycudadecon cudatoolkit=10.2
```

```{list-table}
:header-rows: 1

* - CUDA
  - Linux driver
  - Win driver
* - 10.2
  - ≥ 440.33
  - ≥ 441.22
* - 11.0
  - ≥ 450.36.06
  - ≥ 451.22
* - 11.1
  - ≥ 455.23
  - ≥ 456.38
* - 11.2
  - ≥ 460.27.03
  - ≥ 460.82
```

If you run into trouble, feel free to [open an issue on
github](https://github.com/tlambert03/pycudadecon/issues) and describe your
setup.
