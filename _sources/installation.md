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

* - CUDA toolkit
  - Linux x86_64 driver
  - Win x86_64 driver
* - 11.8
  - ≥ 520.61.05
  - ≥ 522.06
* - 11.7
  - ≥ 515.43.04
  - ≥ 516.01
* - 11.6
  - ≥ 510.39.01
  - ≥ 511.23
* - 11.5
  - ≥ 495.29.05
  - ≥ 496.04
* - 11.4
  - ≥ 470.42.01
  - ≥ 471.11
* - 11.3
  - ≥ 465.19.01
  - ≥ 465.89
* - 11.2
  - ≥ 460.27.03
  - ≥ 460.82
* - 11.1
  - ≥ 455.23
  - ≥ 456.38
* - 11.0
  - ≥ 450.36.06
  - ≥ 451.22
* - 10.2
  - ≥ 440.33
  - ≥ 441.22
```

For the most recent information on GPU driver compatibility, please see the
[NVIDIA CUDA Toolkit Release
Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

If you run into trouble, feel free to [open an issue on
github](https://github.com/tlambert03/pycudadecon/issues) and describe your
setup.
