"""
Input-validation tests for euclid_fast.

Builds small FITS files with astropy and checks that supported files work
and that unsupported ones are rejected with a clear error (instead of
silently giving wrong results).

Run from the project root, in the environment where euclid_fast is installed:

    pip install -r scripts/requirements.txt
    python tests/test_input_validation.py        # plain Python, no pytest needed
    # or:  pip install pytest && pytest tests/
"""
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

import euclid_fast

IMG = np.random.default_rng(0).normal(100, 5, (200, 300)).astype(np.float32)
RA, DEC = 10.0, 20.0


def add_wcs(header, ctype=("RA---TAN", "DEC--TAN"), cd=True):
    header["CTYPE1"], header["CTYPE2"] = ctype
    header["CRVAL1"], header["CRVAL2"] = RA, DEC
    header["CRPIX1"], header["CRPIX2"] = 150, 100
    if cd:
        header["CD1_1"], header["CD1_2"], header["CD2_1"], header["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    else:
        header["CDELT1"], header["CDELT2"] = -1e-4, 1e-4


def write(tmp, name, hdul):
    path = str(Path(tmp) / name)
    hdul.writeto(path)
    return path


def raises(fn, *fragments):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert all(f in msg for f in fragments), f"unexpected message: {msg!r}"
        return
    raise AssertionError("expected a ValueError, none was raised")


def cutout(path):
    return euclid_fast.extract_cutout(path, RA, DEC, 20, 20)


def test_all():
    with tempfile.TemporaryDirectory() as tmp:
        # --- supported ---
        h = fits.PrimaryHDU(IMG); add_wcs(h.header)
        p = write(tmp, "ok.fits", fits.HDUList([h]))
        assert abs(euclid_fast.fits_stats(p).mean - float(IMG.mean())) < 1e-3
        assert len(cutout(p).data) == 400

        h = fits.PrimaryHDU(IMG.astype(np.int16)); add_wcs(h.header)
        h.header["BZERO"], h.header["BSCALE"] = 5.0, 0.5
        p = write(tmp, "int16.fits", fits.HDUList([h]))
        assert abs(euclid_fast.fits_stats(p).mean - (5.0 + 0.5 * float(IMG.astype(np.int16).mean()))) < 1e-3

        # --- rejected: no image in the primary HDU ---
        comp = fits.CompImageHDU(IMG, compression_type="RICE_1")
        p = write(tmp, "compressed.fits", fits.HDUList([fits.PrimaryHDU(), comp]))
        raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 0", "tile-compressed")
        raises(lambda: cutout(p), "NAXIS = 0")

        ext = fits.ImageHDU(IMG); add_wcs(ext.header)
        p = write(tmp, "extension.fits", fits.HDUList([fits.PrimaryHDU(), ext]))
        raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 0", "extension")

        # --- rejected: 3-D cube (used to be read as its first plane only) ---
        h = fits.PrimaryHDU(np.stack([IMG, IMG + 1, IMG + 2])); add_wcs(h.header)
        p = write(tmp, "cube.fits", fits.HDUList([h]))
        raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 3", "2-D")
        raises(lambda: cutout(p), "NAXIS = 3")

        # --- rejected: projections other than plain TAN (used to be treated as TAN) ---
        for name, ctype in (("sin.fits", ("RA---SIN", "DEC--SIN")),
                            ("tansip.fits", ("RA---TAN-SIP", "DEC--TAN-SIP"))):
            h = fits.PrimaryHDU(IMG); add_wcs(h.header, ctype=ctype)
            p = write(tmp, name, fits.HDUList([h]))
            assert euclid_fast.fits_stats(p).naxis1 == 300      # stats need no WCS
            raises(lambda: cutout(p), "unsupported WCS projection", ctype[0])

        h = fits.PrimaryHDU(IMG)                                # no CTYPE at all
        for k, v in (("CRVAL1", RA), ("CRVAL2", DEC), ("CRPIX1", 150), ("CRPIX2", 100),
                     ("CD1_1", -1e-4), ("CD2_2", 1e-4)):
            h.header[k] = v
        p = write(tmp, "noctype.fits", fits.HDUList([h]))
        raises(lambda: cutout(p), "unsupported WCS projection", "<missing>")

        # --- WCS written with CDELT instead of CD, and no WCS: stats ok, cutout rejected ---
        for name, with_cdelt in (("cdelt.fits", True), ("nowcs.fits", False)):
            h = fits.PrimaryHDU(IMG)
            if with_cdelt:
                add_wcs(h.header, cd=False)
            p = write(tmp, name, fits.HDUList([h]))
            assert euclid_fast.fits_stats(p).naxis2 == 200
            raises(lambda: cutout(p), "no WCS header")

        # --- other rejections ---
        h = fits.PrimaryHDU(IMG.astype(np.int64)); add_wcs(h.header)
        p = write(tmp, "int64.fits", fits.HDUList([h]))
        raises(lambda: euclid_fast.fits_stats(p), "BITPIX 64")

        p = str(Path(tmp) / "notfits.fits"); Path(p).write_bytes(b"hello" * 1000)
        raises(lambda: euclid_fast.fits_stats(p), "no END card")


if __name__ == "__main__":
    test_all()
    print("all input-validation checks passed")