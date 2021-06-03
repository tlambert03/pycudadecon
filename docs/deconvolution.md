# Deconvolution

The primary function for performing deconvolution is {func}`~pycudadecon.decon`.

This convenience function is capable of receiving a variety of input types
(filenames, directory names, numpy arrays, or a list of any of those) and
will handle setting up and breaking down the FFT plan on the GPU for all files
being deconvolved.  Keywords arguments will be passed internally to the
{class}`~pycudadecon.RLContext` context manager or the
{func}`~pycudadecon.make_otf` {func}`~pycudadecon.rl_decon` functions.

The setup and breakdown for the GPU-deconvolution can also be performed
manually:

1. call {func}`~pycudadecon.rl_init` with the shape of the raw data and path to
   OTF file.
2. perform deconvolution(s) with {func}`~pycudadecon.rl_decon`.
3. cleanup with {func}`~pycudadecon.rl_cleanup`

As a convenience, the {class}`~pycudadecon.RLContext` context manager will
perform the setup and breakdown automatically:

```python
data = tiffile.imread('some_file.tif') otf = 'path_to_otf.tif' with
RLContext(data.shape, otf) as ctx:
    result = rl_decon(data, output_shape=ctx.out_shape)
```

## API

```{eval-rst}
.. automodule:: pycudadecon
    :members: decon, rl_init, rl_decon,  RLContext, rl_cleanup

.. automodule:: pycudadecon.libcudawrapper
    :members: cuda_reset
```
