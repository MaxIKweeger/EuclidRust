"""
Tests for euclid_fast: supported inputs work, unsupported ones are rejected
with a clear error (instead of silently giving wrong results), and the TPV
distortion agrees with astropy/wcslib.

Everything is built from small synthetic FITS files, so no download is needed.
Run from the project root, in the environment where euclid_fast is installed:

    pip install -r scripts/requirements.txt
    python tests/test_input_validation.py        # plain Python, no pytest needed
    # or:  pip install pytest && pytest tests/
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import euclid_fast

warnings.simplefilter("ignore")

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
        assert all(f in msg for f in fragments), f"unexpected message: {msg!r} (wanted {fragments})"
        return
    raise AssertionError("expected a ValueError, none was raised")


def cutout(path, **kw):
    return euclid_fast.extract_cutout(path, RA, DEC, 20, 20, **kw)


def test_supported_primary(tmp):
    h = fits.PrimaryHDU(IMG); add_wcs(h.header)
    p = write(tmp, "ok.fits", fits.HDUList([h]))
    assert abs(euclid_fast.fits_stats(p).mean - float(IMG.mean())) < 1e-3
    assert len(cutout(p).data) == 400

    h = fits.PrimaryHDU(IMG.astype(np.int16)); add_wcs(h.header)
    h.header["BZERO"], h.header["BSCALE"] = 5.0, 0.5
    p = write(tmp, "int16.fits", fits.HDUList([h]))
    assert abs(euclid_fast.fits_stats(p).mean - (5.0 + 0.5 * float(IMG.astype(np.int16).mean()))) < 1e-3


def test_int64(tmp):
    seg = np.random.default_rng(3).integers(0, 5_000_000, (120, 150)).astype(np.int64)
    p = write(tmp, "seg.fits", fits.HDUList([fits.PrimaryHDU(seg)]))
    s = euclid_fast.fits_stats(p)
    assert (s.min, s.max) == (float(seg.min()), float(seg.max()))
    assert s.mean == float(seg.astype(np.float64).mean())
    assert s.bitpix == 64


def test_extensions(tmp):
    # Layout of Euclid calibrated frames: empty primary, one image per extension.
    sci = fits.ImageHDU(IMG, name="DET11.SCI"); add_wcs(sci.header)
    rms = fits.ImageHDU(IMG * 2, name="DET11.RMS")
    dq = fits.ImageHDU((IMG > 100).astype(np.uint8), name="DET11.DQ")
    p = write(tmp, "frame.fits", fits.HDUList([fits.PrimaryHDU(), sci, rms, dq]))

    infos = euclid_fast.list_hdus(p)
    assert [(i.index, i.name, i.readable) for i in infos] == [
        (0, "", False), (1, "DET11.SCI", True), (2, "DET11.RMS", True), (3, "DET11.DQ", True)]
    assert infos[1].shape == [300, 200] and infos[1].bitpix == -32 and infos[3].bitpix == 8

    for sel, ref in (("DET11.SCI", IMG), (1, IMG), ("det11.rms", IMG * 2), (2, IMG * 2)):
        assert abs(euclid_fast.fits_stats(p, hdu=sel).mean - float(ref.astype(np.float64).mean())) < 1e-3
    assert euclid_fast.fits_stats(p, hdu="DET11.DQ").bitpix == 8

    c = cutout(p, hdu="DET11.SCI")
    x0, y0 = int(round(c.pixel_x_center - 1 - 10)), int(round(c.pixel_y_center - 1 - 10))
    assert np.array_equal(np.array(c.data).reshape(20, 20), IMG[y0:y0 + 20, x0:x0 + 20].astype(np.float64))

    # No hdu given on a file with an empty primary: explain what to pass.
    raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 0", "DET11.SCI", "hdu=")
    raises(lambda: cutout(p), "NAXIS = 0", "hdu=")
    raises(lambda: euclid_fast.fits_stats(p, hdu="NOPE"), "no HDU named 'NOPE'", "DET11.SCI")
    raises(lambda: euclid_fast.fits_stats(p, hdu=9), "no HDU with index 9", "4 HDU")
    # An extension without WCS: stats work, cutouts do not.
    raises(lambda: cutout(p, hdu="DET11.RMS"), "no WCS header")


def test_unreadable_hdus(tmp):
    comp = fits.CompImageHDU(IMG, compression_type="RICE_1", name="SCI")
    p = write(tmp, "compressed.fits", fits.HDUList([fits.PrimaryHDU(), comp]))
    raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 0", "tile-compressed")
    raises(lambda: euclid_fast.fits_stats(p, hdu=1), "tile-compressed image (ZIMAGE)")
    info = euclid_fast.list_hdus(p)[1]
    assert not info.readable and "tile-compressed" in info.note

    table = fits.BinTableHDU.from_columns([fits.Column(name="a", format="E", array=np.arange(5, dtype=np.float32))])
    p = write(tmp, "table.fits", fits.HDUList([fits.PrimaryHDU(), table]))
    raises(lambda: euclid_fast.fits_stats(p, hdu=1), "BINTABLE table, not an image")


def test_rejected_inputs(tmp):
    # 3-D cube (used to be read as its first plane only)
    h = fits.PrimaryHDU(np.stack([IMG, IMG + 1, IMG + 2])); add_wcs(h.header)
    p = write(tmp, "cube.fits", fits.HDUList([h]))
    raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 3", "2-D")
    raises(lambda: cutout(p), "NAXIS = 3")

    # Projections other than TAN / TPV (used to be treated as TAN)
    for name, ctype in (("sin.fits", ("RA---SIN", "DEC--SIN")), ("tansip.fits", ("RA---TAN-SIP", "DEC--TAN-SIP"))):
        h = fits.PrimaryHDU(IMG); add_wcs(h.header, ctype=ctype)
        p = write(tmp, name, fits.HDUList([h]))
        assert euclid_fast.fits_stats(p).naxis1 == 300      # stats need no WCS
        raises(lambda: cutout(p), "unsupported WCS projection", ctype[0])

    h = fits.PrimaryHDU(IMG)                                # no CTYPE at all
    for k, v in (("CRVAL1", RA), ("CRVAL2", DEC), ("CRPIX1", 150), ("CRPIX2", 100), ("CD1_1", -1e-4), ("CD2_2", 1e-4)):
        h.header[k] = v
    p = write(tmp, "noctype.fits", fits.HDUList([h]))
    raises(lambda: cutout(p), "unsupported WCS projection", "<missing>")

    # CDELT instead of CD, and no WCS: stats ok, cutout rejected
    for name, with_cdelt in (("cdelt.fits", True), ("nowcs.fits", False)):
        h = fits.PrimaryHDU(IMG)
        if with_cdelt:
            add_wcs(h.header, cd=False)
        p = write(tmp, name, fits.HDUList([h]))
        assert euclid_fast.fits_stats(p).naxis2 == 200
        raises(lambda: cutout(p), "no WCS header")

    p = str(Path(tmp) / "notfits.fits"); Path(p).write_bytes(b"hello" * 1000)
    raises(lambda: euclid_fast.fits_stats(p), "no END card")
    raises(lambda: euclid_fast.fits_stats(str(Path(tmp) / "missing.fits")), "cannot open")


def test_tpv_matches_astropy(tmp):
    # A strongly distorted TPV header that exercises many polynomial terms:
    # linear, quadratic, cubic, radial (r, r^3, r^5, r^7) and higher orders.
    pv = {"PV1_1": 1.0, "PV1_2": 0.01, "PV1_4": 0.4, "PV1_5": -0.2, "PV1_6": 0.3, "PV1_7": 20.0,
          "PV1_10": -15.0, "PV1_11": 8.0, "PV1_12": 300.0, "PV1_20": 1e4, "PV1_23": 2000.0,
          "PV2_1": 1.0, "PV2_2": -0.01, "PV2_4": 0.3, "PV2_5": 0.25, "PV2_6": -0.4, "PV2_7": 12.0,
          "PV2_10": 18.0, "PV2_11": 6.0, "PV2_16": -250.0, "PV2_30": 1e5, "PV2_39": 1e6}
    img = np.random.default_rng(5).normal(0, 1, (400, 500)).astype(np.float32)
    h = fits.PrimaryHDU(img); add_wcs(h.header, ctype=("RA---TPV", "DEC--TPV"))
    h.header["CRPIX1"], h.header["CRPIX2"] = 250, 200
    for k, v in pv.items():
        h.header[k] = v
    p = write(tmp, "tpv.fits", fits.HDUList([h]))
    wcs = WCS(fits.getheader(p))
    rng = np.random.default_rng(6)
    xs, ys = rng.uniform(30, 470, 200), rng.uniform(30, 370, 200)
    ras, decs = wcs.all_pix2world(xs, ys, 1)
    worst = 0.0
    for x, y, ra, dec in zip(xs, ys, ras, decs):
        c = euclid_fast.extract_cutout(p, float(ra), float(dec), 5, 5)
        worst = max(worst, abs(c.pixel_x_center - x), abs(c.pixel_y_center - y))
    shift = np.hypot(*(np.array(WCS(h.header.copy()).all_world2pix(ras, decs, 1)) - np.array([xs, ys]))).max()
    assert worst < 1e-5, f"TPV disagrees with astropy by up to {worst} px"
    assert shift < 1e-5   # astropy's own round trip is exact, so the reference is sound


def test_all():
    for fn in (test_supported_primary, test_int64, test_extensions, test_unreadable_hdus,
               test_rejected_inputs, test_tpv_matches_astropy):
        with tempfile.TemporaryDirectory() as tmp:
            fn(tmp)
            print("ok  ", fn.__name__)


if __name__ == "__main__":
    test_all()
    print("all checks passed")