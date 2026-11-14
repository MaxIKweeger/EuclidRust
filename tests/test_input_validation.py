"""
Tests for euclid_fast: supported inputs work, unsupported ones are rejected
with a clear error (instead of silently giving wrong results), the TPV
distortion agrees with astropy/wcslib, and tile-compressed images decode to
exactly what astropy/cfitsio decodes.

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
    # A compressed image is readable (see test_compressed_images), but it lives in
    # an extension: without hdu= the error points at it.
    comp = fits.CompImageHDU(IMG, compression_type="RICE_1", name="SCI")
    p = write(tmp, "compressed.fits", fits.HDUList([fits.PrimaryHDU(), comp]))
    raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 0", "tile-compressed RICE_1", "hdu=")
    info = euclid_fast.list_hdus(p)[1]
    assert info.readable and "tile-compressed RICE_1" in info.note and info.shape == [300, 200]

    # Compression algorithms that are not implemented are refused with a clear message.
    hc = fits.CompImageHDU(IMG, compression_type="HCOMPRESS_1", name="SCI")
    p = write(tmp, "hcompress.fits", fits.HDUList([fits.PrimaryHDU(), hc]))
    raises(lambda: euclid_fast.fits_stats(p, hdu=1), "HCOMPRESS_1", "not supported yet")
    info = euclid_fast.list_hdus(p)[1]
    assert not info.readable and "HCOMPRESS_1" in info.note

    table = fits.BinTableHDU.from_columns([fits.Column(name="a", format="E", array=np.arange(5, dtype=np.float32))])
    p = write(tmp, "table.fits", fits.HDUList([fits.PrimaryHDU(), table]))
    raises(lambda: euclid_fast.fits_stats(p, hdu=1), "BINTABLE table, not an image")


def test_rejected_inputs(tmp):
    # 3-D cube (used to be read as its first plane only)
    h = fits.PrimaryHDU(np.stack([IMG, IMG + 1, IMG + 2])); add_wcs(h.header)
    p = write(tmp, "cube.fits", fits.HDUList([h]))
    raises(lambda: euclid_fast.fits_stats(p), "NAXIS = 3", "2-D")
    raises(lambda: cutout(p), "NAXIS = 3")

    # Projections that are not implemented (cylindrical, all-sky...), or a mismatched pair
    for name, ctype in (("car.fits", ("RA---CAR", "DEC--CAR")), ("ait.fits", ("RA---AIT", "DEC--AIT")),
                        ("mixed.fits", ("RA---TAN", "DEC--SIN"))):
        h = fits.PrimaryHDU(IMG); add_wcs(h.header, ctype=ctype)
        p = write(tmp, name, fits.HDUList([h]))
        assert euclid_fast.fits_stats(p).naxis1 == 300      # stats need no WCS
        raises(lambda: cutout(p), "unsupported WCS projection", ctype[0])

    h = fits.PrimaryHDU(IMG)                                # no CTYPE at all
    for k, v in (("CRVAL1", RA), ("CRVAL2", DEC), ("CRPIX1", 150), ("CRPIX2", 100), ("CD1_1", -1e-4), ("CD2_2", 1e-4)):
        h.header[k] = v
    p = write(tmp, "noctype.fits", fits.HDUList([h]))
    raises(lambda: cutout(p), "unsupported WCS projection", "<missing>")

    # No WCS at all: stats ok, cutout rejected
    h = fits.PrimaryHDU(IMG)
    p = write(tmp, "nowcs.fits", fits.HDUList([h]))
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


def _compressed_roundtrip(tmp, name, data, check_mean=True, **kw):
    """Compress with astropy (cfitsio), then check euclid_fast decodes the same pixels."""
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRVAL1"], hdr["CRVAL2"] = RA, DEC
    hdr["CRPIX1"], hdr["CRPIX2"] = data.shape[1] / 2, data.shape[0] / 2
    hdr["CD1_1"], hdr["CD1_2"], hdr["CD2_1"], hdr["CD2_2"] = -2e-4, 0.0, 0.0, 2e-4
    p = write(tmp, name, fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data, header=hdr, **kw)]))
    with fits.open(p) as hl:
        ref = np.array(hl[1].data).astype(np.float64)
        wcs = WCS(hl[1].header)
    s = euclid_fast.fits_stats(p, hdu=1)
    assert (s.min, s.max) == (float(np.nanmin(ref)), float(np.nanmax(ref))), (name, s.min, s.max)
    if check_mean:
        assert abs(s.mean - float(ref.mean())) <= 1e-9 * max(1.0, abs(float(ref.mean()))), name
    ny, nx = ref.shape
    for fx, fy in ((0.3, 0.25), (0.62, 0.5), (0.5, 0.8), (0.85, 0.6)):       # tile borders, off-center
        ra, dec = wcs.all_pix2world(fx * nx, fy * ny, 1)
        c = euclid_fast.extract_cutout(p, float(ra), float(dec), 17, 13, hdu=1)
        x0 = int(round(c.pixel_x_center - 1 - 17 / 2)); y0 = int(round(c.pixel_y_center - 1 - 13 / 2))
        got = np.array(c.data).reshape(13, 17)
        assert np.array_equal(got, ref[y0:y0 + 13, x0:x0 + 17], equal_nan=True), (name, "cutout", fx, fy)


def test_compressed_images(tmp):
    # Tile-compressed images (RICE_1, GZIP_1, GZIP_2), with and without float
    # quantization/dithering, against astropy (which uses cfitsio). The image
    # size is not a multiple of the tile size, to exercise the edge tiles.
    rng = np.random.default_rng(11)
    h, w = 203, 317
    ints = {
        "uint8": rng.integers(0, 255, (h, w)).astype(np.uint8),
        "int16": (rng.normal(0, 400, (h, w)) + np.linspace(-3000, 3000, w)).astype(np.int16),
        "int32": (rng.normal(0, 5e4, (h, w)) + np.linspace(-1e6, 1e6, w)).astype(np.int32),
        "uint16": rng.integers(0, 65535, (h, w)).astype(np.uint16),       # BZERO = 32768
    }
    flt = (rng.normal(100, 7, (h, w)) + 50 * np.sin(np.linspace(0, 20, w))).astype(np.float32)
    flt_nan = flt.copy(); flt_nan[rng.random((h, w)) < 0.03] = np.nan; flt_nan[:20, :30] = np.nan
    n = 0
    for dtype, data in ints.items():
        for algo in ("RICE_1", "GZIP_1", "GZIP_2"):
            _compressed_roundtrip(tmp, f"i{n}.fits", data, compression_type=algo); n += 1
    _compressed_roundtrip(tmp, f"i{n}.fits", ints["int16"], compression_type="RICE_1", tile_shape=(37, 41)); n += 1
    _compressed_roundtrip(tmp, f"i{n}.fits", ints["int32"], compression_type="RICE_1", tile_shape=(64, 64)); n += 1
    for algo in ("RICE_1", "GZIP_1", "GZIP_2"):
        for method in (0, 1, 2):                      # NO_DITHER, SUBTRACTIVE_DITHER_1, _2
            _compressed_roundtrip(tmp, f"f{n}.fits", flt, compression_type=algo, quantize_method=method,
                                  dither_seed=17); n += 1
    for seed in (1, 777, 10000, -1):                  # -1 = checksum-based seed
        _compressed_roundtrip(tmp, f"f{n}.fits", flt, compression_type="RICE_1", quantize_method=1,
                              dither_seed=seed); n += 1
    _compressed_roundtrip(tmp, f"f{n}.fits", flt, compression_type="RICE_1", quantize_method=1,
                          dither_seed=123, tile_shape=(40, 50)); n += 1
    _compressed_roundtrip(tmp, f"f{n}.fits", flt.astype(np.float64), compression_type="RICE_1"); n += 1
    _compressed_roundtrip(tmp, f"f{n}.fits", flt_nan, check_mean=False, compression_type="RICE_1",
                          quantize_method=1, dither_seed=9); n += 1
    for algo in ("GZIP_1", "GZIP_2"):                 # lossless float (no quantization)
        _compressed_roundtrip(tmp, f"f{n}.fits", flt, compression_type=algo, quantize_level=0.0); n += 1
        _compressed_roundtrip(tmp, f"f{n}.fits", flt.astype(np.float64), compression_type=algo,
                              quantize_level=0.0); n += 1


def _wcs_header(code, crval, scale, rot=0.0, form="CD", n=(400, 300), lonpole=None, suffix="", **extra):
    import math
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = f"RA---{code}{suffix}", f"DEC--{code}{suffix}"
    h["CRVAL1"], h["CRVAL2"] = crval
    h["CRPIX1"], h["CRPIX2"] = n[0] / 2 + 0.3, n[1] / 2 - 0.7
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    if form == "CD":
        h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -scale * c, scale * s, scale * s, scale * c
    elif form == "PC":
        h["CDELT1"], h["CDELT2"] = -scale, scale
        h["PC1_1"], h["PC1_2"], h["PC2_1"], h["PC2_2"] = c, -s, s, c
    elif form == "CDELT":
        h["CDELT1"], h["CDELT2"] = -scale, scale
    elif form == "CROTA":
        h["CDELT1"], h["CDELT2"] = -scale, scale
        h["CROTA2"] = rot
    if lonpole is not None:
        h["LONPOLE"] = lonpole
    for k, v in extra.items():
        h[k] = v
    return h


def _check_wcs_roundtrip(tmp, name, header, n=(400, 300), npos=40, margin=10, tol=1e-6, seed=0):
    """Sky positions computed by astropy/wcslib for known pixels must be turned back
    into those pixels by euclid_fast."""
    p = write(tmp, name, fits.HDUList([fits.PrimaryHDU(np.zeros((n[1], n[0]), np.float32), header=header)]))
    wcs = WCS(fits.getheader(p))
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(margin, n[0] - margin, npos), rng.uniform(margin, n[1] - margin, npos)
    ras, decs = wcs.all_pix2world(xs, ys, 1)
    got = euclid_fast.extract_cutouts(p, list(ras), list(decs), 3, 3)
    worst = max(max(abs(c.pixel_x_center - x), abs(c.pixel_y_center - y)) for c, x, y in zip(got, xs, ys))
    assert worst < tol, f"{name}: euclid_fast disagrees with wcslib by up to {worst} px"


def test_wcs_matches_astropy(tmp):
    import itertools
    rng = np.random.default_rng(21)
    k = 0
    # zenithal projections, several fields of view, near the poles and across RA = 0
    crvals = [(150.0, 2.0), (359.95, -45.0), (10.0, 88.0), (200.0, -89.3)]
    for code in ("TAN", "SIN", "ARC", "STG", "ZEA"):
        for crval, scale in itertools.product(crvals, (1e-4, 5e-2)):
            _check_wcs_roundtrip(tmp, f"z{k}.fits", _wcs_header(code, crval, scale, float(rng.uniform(-180, 180))), seed=k); k += 1
    # polynomial zenithal projection
    _check_wcs_roundtrip(tmp, f"z{k}.fits", _wcs_header("ZPN", (120.0, 30.0), 4e-2, 20.0, PV2_1=1.0, PV2_3=-0.08, PV2_5=0.02)); k += 1
    # the other ways of writing the linear matrix
    for form in ("PC", "CDELT", "CROTA"):
        for code in ("TAN", "ARC"):
            rot = 0.0 if form == "CDELT" else float(rng.uniform(-90, 90))
            _check_wcs_roundtrip(tmp, f"m{k}.fits", _wcs_header(code, (30.0, -70.0), 3e-3, rot, form)); k += 1
    # explicit LONPOLE
    for lp in (0.0, 90.0, 180.0, 233.0):
        _check_wcs_roundtrip(tmp, f"l{k}.fits", _wcs_header("SIN", (150.0, -20.0), 2e-3, 15.0, lonpole=lp)); k += 1
    # SIP distortion of a few pixels, several orders, CD and PC/CDELT forms
    for order in (2, 3, 4, 5):
        extra = {"A_ORDER": order, "B_ORDER": order}
        for pp in range(order + 1):
            for q in range(order + 1 - pp):
                if pp + q >= 2:
                    extra[f"A_{pp}_{q}"] = float(rng.normal(0, 4e-6 / 200 ** (pp + q - 2)))
                    extra[f"B_{pp}_{q}"] = float(rng.normal(0, 4e-6 / 200 ** (pp + q - 2)))
        for form in ("CD", "PC"):
            _check_wcs_roundtrip(tmp, f"s{k}.fits", _wcs_header("TAN", (150.0, 2.0), 1e-4, 33.0, form, suffix="-SIP", **extra)); k += 1
    _check_wcs_roundtrip(tmp, f"s{k}.fits", _wcs_header("TAN", (150.0, 2.0), 1e-4, 0.0, suffix="-SIP"))   # -SIP without terms
    # real SIP headers shipped with astropy (HST, PC/CDELT form), when the package has them
    import os
    import astropy.wcs
    data = os.path.join(os.path.dirname(astropy.wcs.__file__), "tests", "data")
    for fname in ("sip.fits", "sip2.fits"):
        path = os.path.join(data, fname)
        if os.path.exists(path):
            src = fits.getheader(path, 0)
            keep = fits.Header()
            for card in src.cards:
                if card.keyword not in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "WCSAXES", "") \
                        and not card.keyword.startswith(("COMMENT", "HISTORY")):
                    keep.append(card)
            _check_wcs_roundtrip(tmp, f"r{k}.fits", keep, n=(256, 256), margin=20, tol=1e-6); k += 1


def _same_stats(a, b):
    return (a.min, a.max, a.mean, a.naxis1, a.naxis2, a.bitpix) == (b.min, b.max, b.mean, b.naxis1, b.naxis2, b.bitpix)


def test_batch(tmp):
    # --- fits_stats_many: several files at once, same results as one at a time ---
    paths = []
    for k in range(6):
        h = fits.PrimaryHDU((IMG + 10 * k).astype(np.float32)); add_wcs(h.header)
        paths.append(write(tmp, f"many{k}.fits", fits.HDUList([h])))
    many = euclid_fast.fits_stats_many(paths)
    assert len(many) == 6 and all(_same_stats(m, euclid_fast.fits_stats(p)) for m, p in zip(many, paths))
    assert euclid_fast.fits_stats_many([]) == []

    missing = str(Path(tmp) / "nope.fits")
    raises(lambda: euclid_fast.fits_stats_many(paths + [missing]), "nope.fits", "cannot open")
    mixed = euclid_fast.fits_stats_many(paths[:2] + [missing] + paths[2:], skip_errors=True)
    assert mixed[2] is None and [m is None for m in mixed].count(True) == 1

    # `hdu` applies to every file (here: files whose images are in an extension)
    ext_paths = []
    for k in range(3):
        sci = fits.ImageHDU((IMG * (k + 1)).astype(np.float32), name="DET11.SCI"); add_wcs(sci.header)
        ext_paths.append(write(tmp, f"ext{k}.fits", fits.HDUList([fits.PrimaryHDU(), sci])))
    res = euclid_fast.fits_stats_many(ext_paths, hdu="DET11.SCI")
    assert [round(r.mean, 2) for r in res] == [round(float(IMG.mean()) * (k + 1), 2) for k in range(3)]
    raises(lambda: euclid_fast.fits_stats_many(ext_paths), "ext0.fits", "NAXIS = 0")

    # --- fits_stats_all_hdus: every image HDU of a frame, in file order ---
    hdus = [fits.PrimaryHDU()]
    for det in ("DET11", "DET12", "DET21"):
        sci = fits.ImageHDU((IMG + len(hdus)).astype(np.float32), name=f"{det}.SCI"); add_wcs(sci.header)
        hdus += [sci, fits.ImageHDU((IMG * 2).astype(np.float32), name=f"{det}.RMS"),
                 fits.ImageHDU((IMG > 100).astype(np.uint8), name=f"{det}.DQ")]
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name="a", format="E", array=np.arange(3, dtype=np.float32))]))
    frame = write(tmp, "frame9.fits", fits.HDUList(hdus))
    allres = euclid_fast.fits_stats_all_hdus(frame)
    assert [r.index for r in allres] == list(range(1, 10))            # primary and table are skipped
    assert [r.name for r in allres[:3]] == ["DET11.SCI", "DET11.RMS", "DET11.DQ"]
    for r in allres:
        s = euclid_fast.fits_stats(frame, hdu=r.index)
        assert (r.min, r.max, r.mean, r.naxis1, r.naxis2, r.bitpix) == (s.min, s.max, s.mean, s.naxis1, s.naxis2, s.bitpix)
    assert [r.name for r in euclid_fast.fits_stats_all_hdus(frame, name_contains="sci")] == \
        ["DET11.SCI", "DET12.SCI", "DET21.SCI"]
    assert euclid_fast.fits_stats_all_hdus(frame, name_contains="nothing-matches") == []

    # --- extract_cutouts: many positions from one file ---
    wcs = WCS(fits.getheader(paths[0]))
    pix = [(60.3, 50.7), (200.0, 120.0), (150.5, 99.5), (5.0, 5.0), (300.0, 190.0), (-50.0, 40.0), (100.0, 60.0)]
    ras, decs = (list(v) for v in wcs.all_pix2world([p[0] for p in pix], [p[1] for p in pix], 1))
    batch = euclid_fast.extract_cutouts(paths[0], ras, decs, 17, 13, skip_errors=True)
    assert len(batch) == len(pix)
    for (ra, dec), b in zip(zip(ras, decs), batch):
        try:
            single = euclid_fast.extract_cutout(paths[0], ra, dec, 17, 13)
        except ValueError:
            assert b is None                                     # outside the image
            continue
        assert b.data == single.data and (b.pixel_x_center, b.pixel_y_center) == (single.pixel_x_center, single.pixel_y_center)
        assert (b.width, b.height) == (17, 13)
    assert any(b is None for b in batch) and sum(b is not None for b in batch) >= 4
    raises(lambda: euclid_fast.extract_cutouts(paths[0], ras, decs, 17, 13), "position", "outside the image")
    raises(lambda: euclid_fast.extract_cutouts(paths[0], ras, decs[:-1], 17, 13), "same length")
    assert euclid_fast.extract_cutouts(paths[0], [], [], 17, 13) == []
    # numpy arrays are accepted, and so are compressed images and extensions
    got = euclid_fast.extract_cutouts(paths[0], np.array(ras[:3]), np.array(decs[:3]), 5, 5)
    assert len(got) == 3
    comp = fits.CompImageHDU(IMG, header=fits.getheader(paths[0]), compression_type="RICE_1")
    cp = write(tmp, "batch_comp.fits", fits.HDUList([fits.PrimaryHDU(), comp]))
    cb = euclid_fast.extract_cutouts(cp, ras, decs, 17, 13, hdu=1, skip_errors=True)
    for (ra, dec), b in zip(zip(ras, decs), cb):
        try:
            single = euclid_fast.extract_cutout(cp, ra, dec, 17, 13, hdu=1)
        except ValueError:
            assert b is None
            continue
        assert b.data == single.data


def test_all():
    for fn in (test_supported_primary, test_int64, test_extensions, test_unreadable_hdus,
               test_rejected_inputs, test_tpv_matches_astropy, test_wcs_matches_astropy,
               test_compressed_images, test_batch):
        with tempfile.TemporaryDirectory() as tmp:
            fn(tmp)
            print("ok  ", fn.__name__)


if __name__ == "__main__":
    test_all()
    print("all checks passed")