# Command Line Interface

If you installed `pycudadecon` using conda, the original binaries for OTF
generation (`radialft`) and deconvolution (`cudadecon`) will also be
installed in your conda environment. The CLI is not created by pycudadecon;
it is entirely defined by [cudadecon](https://github.com/scopetools/cudadecon).

## cudadecon

The `cudadecon` command runs deconvolution (with deskewing and rotation if
desired) on all of the files in the `--input-dir` whose names match the
pattern `--filename-pattern`, using the OTF specified by `--otf-arg`.

### Examples

```bash
# deconvolve a folder of images
cudadecon /folder/of/images 488nm /path/to/488nm_otf.tif -z 0.3

# a typical lattice experiment might also add the deskew flag and maybe MIPs
cudadecon /folder/of/images 488nm /path/to/488nm_otf.tif -z 0.3 -D 31.5 -M 0 0 1
```

Run `cudadecon --help` at the command prompt for the full menu of options.
(the output below may be slightly out of date)

```text
$ cudadecon -h

cudaDeconv.  Version: 0.7.0
:
  --input-dir arg                   Folder of input images
  --filename-pattern arg            File name pattern to find input images
                                    to process
  --otf-file arg                    OTF file
  --drdata arg (=0.104)             Image x-y pixel size (um)
  -z [ --dzdata ] arg (=0.25)       Image z step (um)
  --drpsf arg (=0.104)              PSF x-y pixel size (um)
  -Z [ --dzpsf ] arg (=0.1)         PSF z step (um)
  -l [ --wavelength ] arg (=0.525)  Emission wavelength (um)
  -W [ --wiener ] arg (=-1)         Wiener constant (regularization
                                    factor); if this value is positive then
                                    do Wiener filter instead of R-L
  -b [ --background ] arg (=90)     User-supplied background
  -e [ --napodize ] arg (=15)       # of pixels to soften edge with
  -E [ --nzblend ] arg (=0)         # of top and bottom sections to blend
                                    in to reduce axial ringing
  -d [ --dupRevStack ]              Duplicate reversed stack prior to decon
                                    to reduce Z ringing
  -n [ --NA ] arg (=1.2)            Numerical aperture
  -i [ --RL ] arg (=15)             Run Richardson-Lucy, and set how many
                                    iterations
  -D [ --deskew ] arg (=0)          Deskew angle; if not 0.0 then perform
                                    deskewing before deconv
  --dcbds                           If deskewing, do it after decon;
                                    require sample-scan PSF and non-RA 3D OTF
  --padval arg (=0)                 Value to pad image with when deskewing
  -w [ --width ] arg (=0)           If deskewed, the output image's width
  -x [ --shift ] arg (=0)           If deskewed, the output image's extra
                                    shift in X (positive->left
  -R [ --rotate ] arg (=0)          Rotation angle; if not 0.0 then perform
                                    rotation around y axis after deconv
  -S [ --saveDeskewedRaw ]          Save deskewed raw data to files
  -C [ --crop ] arg                 Crop final image size to [x1:x2, y1:y2,
                                    z1:z2]; takes 6 integers separated by
                                    space: x1 x2 y1 y2 z1 z2;
  -M [ --MIP ] arg                  Save max-intensity projection along x,
                                    y, or z axis; takes 3 binary numbers
                                    separated by space: 0 0 1
  -m [ --rMIP ] arg                 Save max-intensity projection of raw
                                    deskewed data along x, y, or z axis;
                                    takes 3 binary numbers separated by
                                    space: 0 0 1
  -u [ --uint16 ]                   Save result in uint16 format; should be
                                    used only if no actual decon is
                                    performed
  -a [ --DoNotAdjustResForFFT ]     Don't change data resolution size.
                                    Otherwise data is cropped to perform
                                    faster, more memory efficient FFT: size
                                    factorable into 2,3,5,7)
  --Pad arg (=0)                    Pad the image data with mirrored values
                                    to avoid edge artifacts. Currently only
                                    enabled when rotate and deskew are
                                    zero.
  --LSC arg                         Lightsheet correction file
  --FlatStart                       Start the RL from a guess that is a
                                    flat image filled with the median image
                                    value.  This may suppress noise.
  -p [ --bleachCorrection ]         Apply bleach correction when running
                                    multiple images in a single batch
  --lzw                             Use LZW tiff compression
  --skip arg (=0)                   Skip the first 'skip' number of files.
  --no_overwrite                    Don't reprocess files that are already
                                    deconvolved (i.e. exist in the GPUdecon
                                    folder).
  -Q [ --DevQuery ]                 Show info and indices of available GPUs
  -h [ --help ]                     This help message.
  -v [ --version ]                  show version and quit
```

## radialft

The `radialft` command turns a 3D PSF volume into a radially-averaged 2D
complex OTF file that can be used by `cudaDecon` (or a 3D OTF file if
--3Dout is specified).

### Examples

```bash
radialft /path/to/psf_file.tif /path/to/new_otf_file.tif --fixorigin 10 --nocleanup
```

Run `radialft --help` at the command prompt for the full menu of options

```text
$ radialft --help

    --input-file arg           input PSF file
    --output-file arg          output OTF file to write
    --na arg (=1.25)           NA of detection objective
    --nimm arg (=1.29999995)   refractive index of immersion medium
    --xyres arg (=0.104000002) x-y pixel size
    --zres arg (=0.104000002)  z pixel size
    --wavelength arg (=530)    emission wavelength in nm
    --fixorigin arg (=5)       for all kz, extrapolate using pixels kr=1 to this
                                pixel to get value for kr=0
    --krmax arg (=0)           pixels outside this limit will be zeroed
                                (overwriting estimated value from NA and NIMM)
    --nocleanup                elect not to do clean-up outside OTF support
    --background arg           use user-supplied background instead of the
                                estimated
    -3 [ --3Dout ]             Output 3D, instead of rotationally averaged, OTF
    -h [ --help ]              produce help message
```
