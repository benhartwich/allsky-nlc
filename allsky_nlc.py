""" allsky_nlc.py

Noctilucent Cloud (NLC / "Leuchtende Nachtwolken") candidate detector for Allsky.
https://github.com/AllskyTeam/allsky

Author:      Benjamin Hartwich (https://astronomy.garden)
Home / docs: https://github.com/benhartwich/allsky-nlc

NLCs are mesospheric clouds at ~76-85 km. They only become visible in deep
twilight, when the observer is already in darkness but the clouds — far higher
than tropospheric weather — are still lit by the Sun below the horizon. That
geometry pins down exactly when and where to look:

  * WHEN:  the Sun is roughly 6-16 deg below the horizon (nautical to the start
           of astronomical twilight). Brighter than -6 the twilight sky drowns
           them; darker than -16 the clouds themselves fall into Earth's shadow.
  * WHERE: low above the horizon, toward the Sun's below-horizon azimuth
           (in a Central-European summer: NW-N at dusk, N-NE at dawn).

This module runs on both the day and night flows and gates itself internally on
the Sun's altitude (computed with ephem), so it fires only inside the twilight
window regardless of where Allsky puts the day/night switch. Inside the window
it builds a region of interest as a low-altitude band centred on the Sun's
azimuth — using the fisheye calibration (calibration.json) for an accurate,
Sun-following band, or a simple horizon band as a graceful fallback.

Within that band a NLC candidate is a feature that is, all at once:
  * BRIGHT relative to the smooth twilight gradient (structured, not the arch),
  * BLUE (electric blue / silvery-white: blue channel well above red — this
    rejects the orange twilight glow near the Sun and orange light-pollution
    domes), and
  * STRUCTURED (fine bands / whirls give real high-frequency texture, unlike
    smooth clear twilight or dark tropospheric overcast, which is in shadow).

Honest scope: reliably separating NLC from twilight-lit thin cirrus is genuinely
hard to fully automate. This is a *candidate* detector — it computes an index,
flags likely nights, and (optionally) saves a thumbnail of the band so a human
can confirm. Pair it with allsky_skyalert for a "possible NLC now" push. Results
are written to a rolling nlc.json for the dashboard.

Field-of-view caveat: a strongly zoomed all-sky lens may not image the sky right
down to the horizon (its fisheye circle can be larger than the sensor). NLC live
low, so with such a lens only the UPPER part of a strong display — the part that
climbs above the frame's lowest visible altitude — will be caught, and only in
the azimuths the frame actually reaches. The band mask is always clipped to the
visible frame, so the index is computed over whatever sky is genuinely in view.
"""
import allsky_shared as s
import os
import json
import math
import time
import subprocess
import cv2
import numpy as np

metaData = {
    "name": "Noctilucent Cloud Detector",
    "description": "Flags possible noctilucent clouds (NLC) in the twilight sky (sun-elevation gated, sunward horizon band)",
    "version": "v0.1.1",
    "events": [
        "day",
        "night"
    ],
    "experimental": "true",
    "module": "allsky_nlc",
    "arguments": {
        "sun_lo": "-16",
        "sun_hi": "-9",
        "alt_lo": "12",
        "alt_hi": "45",
        "az_half": "75",
        "blue_excess": "8",
        "residual_thr": "10",
        "dark_floor": "30",
        "bright_ceil": "245",
        "edge_erode": "8",
        "min_index": "0.4",
        "work_width": "720",
        "mask": "",
        "calibration": "calibration.json",
        "history_hours": "72",
        "save_thumbnail": "true",
        "publish_web": "true",
        "debug": "false"
    },
    "argumentdetails": {
        "sun_lo": {
            "required": "false",
            "description": "Sun altitude — dark limit (deg)",
            "help": "Only look while the Sun is at or above this altitude. Below it, NLCs themselves fall into Earth's shadow. Negative number (default -16).",
            "type": {"fieldtype": "spinner", "min": -20, "max": -8, "step": 1}
        },
        "sun_hi": {
            "required": "false",
            "description": "Sun altitude — bright limit (deg)",
            "help": "Only look while the Sun is at or below this altitude. Above it the twilight sky is too bright (on a zoomed lens the low sunward sky stays blown-out until fairly dark, hence -9 by default). Negative number.",
            "type": {"fieldtype": "spinner", "min": -14, "max": -3, "step": 1}
        },
        "alt_lo": {
            "required": "false",
            "description": "ROI band — bottom altitude (deg)",
            "help": "Lower edge of the search band. NOTE: many all-sky lenses (incl. this one) do not image below ~20-25 deg toward most azimuths, so a lower value simply gets clipped to what the sensor sees. NLC near the true horizon may be out of frame entirely.",
            "type": {"fieldtype": "spinner", "min": 0, "max": 30, "step": 1}
        },
        "alt_hi": {
            "required": "false",
            "description": "ROI band — top altitude (deg)",
            "help": "Upper edge of the search band. NLCs are usually low but reach ~30-40 deg in a strong display — which is the part a zoomed all-sky lens can actually catch.",
            "type": {"fieldtype": "spinner", "min": 15, "max": 70, "step": 1}
        },
        "az_half": {
            "required": "false",
            "description": "ROI band — azimuth half-width (deg)",
            "help": "Half-width of the band either side of the Sun's azimuth. 75 = a 150 deg arc centred sunward.",
            "type": {"fieldtype": "spinner", "min": 20, "max": 120, "step": 5}
        },
        "blue_excess": {
            "required": "false",
            "description": "Min blue excess (B-R)",
            "help": "A NLC pixel must have blue clearly above red (electric blue). Rejects the orange twilight glow and orange light-pollution domes. Higher = stricter.",
            "type": {"fieldtype": "spinner", "min": 0, "max": 40, "step": 1}
        },
        "residual_thr": {
            "required": "false",
            "description": "Min brightness residual",
            "help": "How far above the smoothed twilight background a feature must sit (removes the smooth twilight arch, keeps structured cloud). Higher = stricter.",
            "type": {"fieldtype": "spinner", "min": 4, "max": 60, "step": 1}
        },
        "dark_floor": {
            "required": "false",
            "description": "Dark floor (brightness)",
            "help": "Pixels dimmer than this are treated as shadowed tropospheric cloud / empty sky, never NLC.",
            "type": {"fieldtype": "spinner", "min": 0, "max": 120, "step": 5}
        },
        "bright_ceil": {
            "required": "false",
            "description": "Bright ceiling (saturation)",
            "help": "Pixels brighter than this are treated as blown-out twilight or burnt-in overlay graphics, never NLC. NLC are bright but not sensor-saturated. Run this module BEFORE allsky_overlay so the compass/text are not in the image.",
            "type": {"fieldtype": "spinner", "min": 150, "max": 255, "step": 5}
        },
        "edge_erode": {
            "required": "false",
            "description": "Edge erosion (px)",
            "help": "Shrink the bright-sky region inward by this many pixels (at analysis scale) before looking for structure. Removes the false 'structure' at the bright/dark boundary — the vignette rim, horizon and treeline. Higher = safer but ignores more of the lowest sky. Set a sky mask too for best results.",
            "type": {"fieldtype": "spinner", "min": 0, "max": 25, "step": 1}
        },
        "min_index": {
            "required": "false",
            "description": "Detection threshold (% of band)",
            "help": "Flag a NLC candidate when at least this percentage of the band is blue+bright+structured.",
            "type": {"fieldtype": "spinner", "min": 0.1, "max": 20, "step": 0.1}
        },
        "work_width": {
            "required": "false",
            "description": "Analysis width (px)",
            "help": "The band crop is downscaled to this width for speed. 720 is plenty for structure detection on a 4K frame.",
            "type": {"fieldtype": "spinner", "min": 320, "max": 1920, "step": 40}
        },
        "mask": {
            "required": "false",
            "description": "Sky mask (optional)",
            "help": "Optional mask image (overlay images folder). Black = ignore (e.g. trees on the horizon). Intersected with the band.",
            "type": {"fieldtype": "image"}
        },
        "calibration": {
            "required": "false",
            "description": "Fisheye calibration file",
            "help": "calibration.json in the module folder. With it the band follows the Sun's azimuth exactly; without it a generic horizon band is used.",
            "type": {"fieldtype": "text"}
        },
        "history_hours": {
            "required": "false",
            "description": "History (hours)",
            "help": "How much history to keep in nlc.json for charting.",
            "type": {"fieldtype": "spinner", "min": 1, "max": 336, "step": 1}
        },
        "save_thumbnail": {
            "required": "false",
            "description": "Save candidate thumbnail",
            "help": "When a candidate is flagged, save a colour crop of the band so you can confirm it by eye.",
            "type": {"fieldtype": "checkbox"}
        },
        "publish_web": {
            "required": "false",
            "description": "Publish to Website",
            "help": "Copy nlc.json into the website folder (and upload to the remote website if enabled) so the dashboard can read it.",
            "type": {"fieldtype": "checkbox"}
        },
        "debug": {
            "required": "false",
            "description": "Enable debug images",
            "help": "Write the band ROI and candidate mask to the allsky tmp debug folder.",
            "tab": "Debug",
            "type": {"fieldtype": "checkbox"}
        }
    },
    "changelog": {
        "v0.1.0": [
            {
                "author": "Benjamin Hartwich",
                "authorurl": "https://astronomy.garden",
                "changes": "Initial sun-gated, sunward-band noctilucent cloud candidate detector (blue+bright+structured), rolling json + thumbnail for confirmation"
            }
        ],
        "v0.1.1": [
            {
                "author": "Benjamin Hartwich",
                "authorurl": "https://astronomy.garden",
                "changes": "Also upload the candidate thumbnail to the remote website (nlc/ subfolder) so the dashboard banner's image link works remotely"
            }
        ]
    }
}

_maskCache = {"name": None, "mask": None}
_calibCache = {"name": None, "calib": None}


def _truthy(v):
    """Checkbox args arrive from the flow config as the STRING 'true'/'false';
    'false' is truthy in Python, so parse booleans explicitly."""
    return v is True or (not isinstance(v, bool) and str(v).strip().lower() in ("true", "1", "yes", "on"))


# --- Sun position -----------------------------------------------------------

def _sunAltAz():
    """(altitude_deg, azimuth_deg) of the Sun. Prefers Allsky's own values from
    the capture environment, else computes them with ephem from the configured
    lat/lon. Returns (None, None) if neither is available."""
    alt_s = s.getEnvironmentVariable("AS_SUN_ALTITUDE")
    az_s = s.getEnvironmentVariable("AS_SUN_AZIMUTH")
    if alt_s not in (None, "") and az_s not in (None, ""):
        try:
            return float(alt_s), float(az_s)
        except (TypeError, ValueError):
            pass
    try:
        import ephem
        obs = ephem.Observer()
        obs.lat = str(s.convertLatLon(s.getSetting("latitude")))
        obs.lon = str(s.convertLatLon(s.getSetting("longitude")))
        sun = ephem.Sun()
        sun.compute(obs)
        return math.degrees(float(sun.alt)), math.degrees(float(sun.az))
    except Exception:
        return None, None


# --- Region of interest -----------------------------------------------------

def _loadCalib(name):
    if _calibCache["name"] == name and _calibCache["calib"] is not None:
        return _calibCache["calib"]
    path = name if os.path.isabs(name) else os.path.join(os.path.dirname(__file__), name)
    try:
        with open(path) as fh:
            c = json.load(fh)
        _calibCache.update(name=name, calib=c)
        return c
    except Exception:
        _calibCache.update(name=name, calib=None)
        return None


def _altaz_to_pixel(alt, az, c):
    """(altitude_deg, azimuth_deg) -> image pixel (x, y) for the cubic-equidistant
    fisheye model in calibration.json (same maths as allsky_fisheye)."""
    t = (90.0 - alt) / 90.0
    r = c["a1"] * t + c["a3"] * t ** 3
    ang = math.radians(c["rot_deg"] + c["flip"] * az)
    return c["cx"] + r * math.sin(ang), c["cy"] - r * math.cos(ang)


def _bandMask(shape, sun_az, params, calib):
    """uint8 mask of the low-altitude band centred on the Sun's azimuth.

    With calibration: a true (alt,az) band swept sunward. Without calibration:
    a generic outer/upper horizon annulus (best-effort, orientation unknown)."""
    h, w = shape
    m = np.zeros(shape, np.uint8)
    alt_lo = s.asfloat(params.get("alt_lo", 2))
    alt_hi = s.asfloat(params.get("alt_hi", 25))
    az_half = s.asfloat(params.get("az_half", 75))

    if calib is not None:
        azs = np.linspace(sun_az - az_half, sun_az + az_half, 121)
        lo = [_altaz_to_pixel(alt_lo, a, calib) for a in azs]
        hi = [_altaz_to_pixel(alt_hi, a, calib) for a in azs[::-1]]
        poly = np.array(lo + hi, np.float32)
        poly = poly[np.isfinite(poly).all(axis=1)]
        if len(poly) >= 3:
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
            return m
    # fallback: outer horizon annulus, upper half (image is North-up, sunward
    # NW/NE both sit in the upper half). Deliberately generous.
    cx, cy = w // 2, h // 2
    rad = min(cx, cy)
    r_out = int(rad * (1.0 - alt_lo / 90.0))
    r_in = int(rad * (1.0 - alt_hi / 90.0))
    cv2.circle(m, (cx, cy), r_out, 255, -1)
    cv2.circle(m, (cx, cy), r_in, 0, -1)
    m[cy:, :] = 0
    return m


def _userMask(params, shape):
    """Optional tree/horizon mask (white = keep). Cached."""
    name = params.get("mask", "").strip()
    if not name:
        return None
    if _maskCache["name"] == name and _maskCache["mask"] is not None \
            and _maskCache["mask"].shape == shape:
        return _maskCache["mask"]
    p = os.path.join(s.ALLSKY_OVERLAY, "images", name)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    _maskCache.update(name=name, mask=m)
    return m


# --- Detection --------------------------------------------------------------

def _detect(bgr, band, params):
    """Score the band for NLC. Returns a dict of metrics + a candidate flag.

    Works on a downscaled crop of the band's bounding box for speed. NLC pixels
    are BRIGHT above the smooth twilight background AND BLUE AND STRUCTURED."""
    ys, xs = np.where(band > 0)
    if len(ys) < 50:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    crop = bgr[y0:y1, x0:x1]
    bmask = band[y0:y1, x0:x1]

    work_w = max(160, s.int(params.get("work_width", 720)))
    ch, cw = crop.shape[:2]
    if cw > work_w:
        scale = work_w / float(cw)
        crop = cv2.resize(crop, (work_w, max(1, int(ch * scale))), interpolation=cv2.INTER_AREA)
        bmask = cv2.resize(bmask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    inb = bmask > 127
    n_band = int(inb.sum())
    if n_band < 50:
        return None

    # suppress stars: they are bright, bluish point sources that would otherwise
    # look exactly like tiny NLC. A median filter removes point sources while
    # preserving the extended structure of a real cloud.
    crop = cv2.medianBlur(crop, 5)

    b, g, r = cv2.split(crop.astype(np.int16))
    val = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    res_thr = s.asfloat(params.get("residual_thr", 12))
    be_thr = s.asfloat(params.get("blue_excess", 8))
    floor = s.asfloat(params.get("dark_floor", 30))
    ceil = s.asfloat(params.get("bright_ceil", 245))
    erode_px = max(1, s.int(params.get("edge_erode", 8)))

    # INTERIOR sky only. The single hardest false positive is the bright edge
    # where the glowing low sky meets the dark vignette / horizon / trees: it is
    # bright, blue and (broken up by the treeline) looks "structured", yet it is
    # just an intensity step, not sky. Eroding the bright, in-band region inward
    # removes every such bright/dark boundary in one stroke — vignette rim AND
    # tree edges — leaving only genuine interior sky where real NLC ripples live.
    valid = (inb & (val > floor) & (val < ceil)).astype(np.uint8) * 255
    valid = cv2.erode(valid, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    interior = valid > 0
    n_int = int(interior.sum())
    if n_int < 50:
        return None

    # remove the smooth twilight glow -> structured brightening only. The kernel
    # is tied to the analysis width (~1/12) so it flattens the broad glow but
    # keeps NLC-scale ripples; a larger kernel would let the glow leak through.
    k = max(31, (crop.shape[1] // 12) | 1)
    background = cv2.GaussianBlur(val, (k, k), 0)
    residual = val - background
    blue_excess = b - r                                  # electric-blue signature

    cand = interior & (residual > res_thr) & (blue_excess > be_thr)
    cand_u8 = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # keep only coherent, EXTENDED blobs — a real display is patchy/banded, not
    # single specks (which would be residual stars or hot pixels)
    keep = np.zeros_like(cand_u8)
    n_blobs = 0
    if int(cand_u8.sum()):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, 8)
        min_blob = max(25, n_int // 150)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob:
                keep[labels == i] = 255
                n_blobs += 1
    kept = keep > 0
    n_cand = int(kept.sum())
    index = 100.0 * n_cand / n_int
    structure = float(residual[kept].mean()) if n_cand else 0.0
    blue_mean = float(blue_excess[kept].mean()) if n_cand else 0.0
    cand_u8 = keep

    min_index = s.asfloat(params.get("min_index", 0.4))
    candidate = (index >= min_index) and (n_blobs >= 1)

    return {
        "index": round(index, 2),
        "structure": round(structure, 1),
        "blue": round(blue_mean, 1),
        "blobs": n_blobs,
        "candidate": bool(candidate),
        "crop_box": (x0, y0, x1, y1),
        "cand_mask": cand_u8,
    }


# --- Website / history (same pattern as allsky_skyquality) ------------------

def _websiteDataDir():
    website = s.getEnvironmentVariable("ALLSKY_WEBSITE")
    if not website:
        website = os.path.join(s.getEnvironmentVariable("ALLSKY_HOME") or os.path.expanduser("~/allsky"),
                               "html", "allsky")
    return website


def _uploadRemote(local, fname, subdir=""):
    """Upload a file to the remote website (optionally into a subdir, e.g. 'nlc').
    The subdir must already exist on the server (upload.sh does not create it).
    Never raises."""
    try:
        if str(s.getSetting("useremotewebsite")).lower() not in ("true", "1", "yes", "on"):
            return
        scripts = s.getEnvironmentVariable("ALLSKY_SCRIPTS") or \
            os.path.join(s.getEnvironmentVariable("ALLSKY_HOME") or os.path.expanduser("~/allsky"), "scripts")
        uploader = os.path.join(scripts, "upload.sh")
        if not os.path.isfile(uploader) or not os.path.isfile(local):
            return
        rdir = (s.getSetting("remotewebsiteimagedir") or "").rstrip("/")
        if subdir:
            rdir = f"{rdir}/{subdir.strip('/')}"
        subprocess.Popen([uploader, "--silent", "--wait", "--remote-web", local, rdir, fname, "NLC"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as ex:
        s.log(1, f"WARNING: nlc remote upload failed: {ex}")


def _appendHistory(record, hours, publish_web):
    path = os.path.join(s.ALLSKY_TMP, "nlc.json")
    try:
        data = json.load(open(path)) if os.path.exists(path) else []
    except Exception:
        data = []
    data.append(record)
    cutoff = record["t"] - hours * 3600
    data = [d for d in data if d.get("t", 0) >= cutoff][-5000:]
    try:
        json.dump(data, open(path, "w"))
    except Exception as ex:
        s.log(1, f"WARNING: nlc could not write history: {ex}")
        return
    if publish_web:
        try:
            ddir = _websiteDataDir()
            os.makedirs(ddir, exist_ok=True)
            webpath = os.path.join(ddir, "nlc.json")
            json.dump(data, open(webpath, "w"))
            _uploadRemote(webpath, "nlc.json")
        except Exception as ex:
            s.log(1, f"WARNING: nlc could not publish to website: {ex}")


def _saveThumb(bgr, box, publish_web):
    """Save a colour crop of the band for human confirmation. Returns filename."""
    try:
        x0, y0, x1, y1 = box
        pad = 20
        h, w = bgr.shape[:2]
        crop = bgr[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if crop.size == 0:
            return None
        fname = time.strftime("nlc-%Y%m%d_%H%M%S.jpg", time.localtime())
        tmpdir = os.path.join(s.ALLSKY_TMP, "nlc")
        os.makedirs(tmpdir, exist_ok=True)
        cv2.imwrite(os.path.join(tmpdir, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if publish_web:
            wdir = os.path.join(_websiteDataDir(), "nlc")
            os.makedirs(wdir, exist_ok=True)
            webthumb = os.path.join(wdir, fname)
            cv2.imwrite(webthumb, crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            _uploadRemote(webthumb, fname, subdir="nlc")     # -> remote <root>/nlc/<fname>
        return fname
    except Exception as ex:
        s.log(1, f"WARNING: nlc could not save thumbnail: {ex}")
        return None


def _compass(az):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((az % 360) / 22.5 + 0.5) % 16]


def nlc(params, event):
    if s.image is None:
        return "No image available"
    if len(s.image.shape) != 3:
        return "NLC needs a colour image"

    debug = _truthy(params.get("debug", False))

    sun_alt, sun_az = _sunAltAz()
    if sun_alt is None:
        s.setEnvironmentVariable("AS_NLC", "0")
        return "NLC: no Sun position (need ephem or AS_SUN_ALTITUDE) — skipped"

    sun_lo = s.asfloat(params.get("sun_lo", -16))
    sun_hi = s.asfloat(params.get("sun_hi", -6))
    s.setEnvironmentVariable("AS_NLC_SUNALT", f"{sun_alt:.1f}")
    if not (sun_lo <= sun_alt <= sun_hi):
        s.setEnvironmentVariable("AS_NLC", "0")
        s.setEnvironmentVariable("AS_NLC_INDEX", "0")
        return f"NLC: outside twilight window (sun {sun_alt:.1f} deg, need {sun_lo}..{sun_hi}) — skipped"

    calib = _loadCalib(params.get("calibration", "calibration.json").strip() or "calibration.json")
    shape = s.image.shape[:2]
    band = _bandMask(shape, sun_az, params, calib)
    umask = _userMask(params, shape)
    if umask is not None:
        band = cv2.bitwise_and(band, umask)

    if debug:
        s.startModuleDebug(metaData["module"])
        s.writeDebugImage(metaData["module"], "nlc-band.png",
                          cv2.bitwise_and(s.image, s.image, mask=band))

    det = _detect(s.image, band, params)
    if det is None:
        s.setEnvironmentVariable("AS_NLC", "0")
        s.setEnvironmentVariable("AS_NLC_INDEX", "0")
        return f"NLC: band empty/too small (sun {sun_alt:.1f} deg) — nothing to score"

    if debug:
        s.writeDebugImage(metaData["module"], "nlc-candidate.png", det["cand_mask"])

    thumb = None
    if det["candidate"] and _truthy(params.get("save_thumbnail", True)):
        thumb = _saveThumb(s.image, det["crop_box"], _truthy(params.get("publish_web", True)))

    s.setEnvironmentVariable("AS_NLC", "1" if det["candidate"] else "0")
    s.setEnvironmentVariable("AS_NLC_INDEX", f"{det['index']:.2f}")
    s.setEnvironmentVariable("AS_NLC_SUNAZ", f"{sun_az:.1f}")
    s.setEnvironmentVariable("AS_NLC_DIR", _compass(sun_az))

    rec = {
        "t": int(time.time()),
        "sun_alt": round(sun_alt, 1),
        "sun_az": round(sun_az, 1),
        "dir": _compass(sun_az),
        "index": det["index"],
        "structure": det["structure"],
        "blue": det["blue"],
        "blobs": det["blobs"],
        "nlc": det["candidate"],
        "calib": calib is not None,
    }
    if thumb:
        rec["thumb"] = thumb
    _appendHistory(rec, s.int(params.get("history_hours", 72)), _truthy(params.get("publish_web", True)))

    if det["candidate"]:
        result = (f"POSSIBLE NLC — index {det['index']:.1f}% of band, {det['blobs']} patches, "
                  f"structure {det['structure']:.0f}, blue +{det['blue']:.0f}, "
                  f"low in the {_compass(sun_az)} (sun {sun_alt:.1f} deg)")
        s.log(1, f"INFO: {result}")
    else:
        result = (f"NLC: clear — index {det['index']:.2f}% (sun {sun_alt:.1f} deg, "
                  f"watching the {_compass(sun_az)} horizon)")
        s.log(4, f"INFO: {result}")
    return result


def nlc_cleanup():
    moduleData = {
        "metaData": metaData,
        "cleanup": {
            "files": {os.path.join(s.ALLSKY_TMP, "nlc.json")},
            "env": {"AS_NLC", "AS_NLC_INDEX", "AS_NLC_SUNALT", "AS_NLC_SUNAZ", "AS_NLC_DIR"}
        }
    }
    s.cleanupModule(moduleData)
