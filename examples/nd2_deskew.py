# pip install pims_nd2
import os

import numpy as np
from pims import ND2_Reader
from skimage.io import imsave

from pycudadecon.affine import affineGPU

# needed to flip the sign on the transform
DEFAULT_TMAT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.70711, 0, 1, 0], [0, 0, 0, 1]])


def deskew_file(path, tmat=DEFAULT_TMAT, mip=True):
    """Deskews an nd2 file at `path` using matrix tmat.

    Will create a new folder (with the same name as the file)
    of multichannel deskewed files for each timepoint.

    Args:
        path (str): the nd2 file to process
        tmat (np.ndarray): the transformation matrix
        mip (bool): Whether to write a MIP file
    """
    tmat = np.array(tmat)
    dirname, fname = os.path.split(path)
    fname, ext = os.path.splitext(fname)
    outdir = os.path.join(dirname, fname + "_deskewed")
    os.makedirs(outdir, exist_ok=True)
    with ND2_Reader(path) as frames:
        frames.bundle_axes = "czyx"
        for frame in frames:
            print(f"processing frame: {frame.frame_no + 1:4} of {len(frames)}")

            # do the actual deskew to each channel
            # the .T rotates it 90 degrees
            out = np.stack([affineGPU(chan, tmat).T for chan in frame])
            dst = os.path.join(outdir, f"{fname}_{frame.frame_no:04}.tif")

            # could add metadata here for voxel sizes
            imsave(dst, np.transpose(out, (1, 0, 2, 3)), imagej=True)
            if mip:
                mipdir = os.path.join(outdir, "MIPs")
                os.makedirs(mipdir, exist_ok=True)
                dst = os.path.join(mipdir, f"{fname}_{frame.frame_no:04}_mip.tif")
                imsave(dst, out.max(1), imagej=True)


if __name__ == "__main__":
    import sys

    deskew_file(sys.argv[1])
