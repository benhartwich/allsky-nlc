# allsky_nlc

Noctilucent-cloud (**NLC** / *Leuchtende Nachtwolken*) candidate detector for
[Allsky](https://github.com/AllskyTeam/allsky).

NLCs are the highest clouds in the atmosphere — thin sheets of ice at ~76–85 km,
far above the weather. They only light up in **deep twilight**, when you are
already in darkness but the clouds, up in the mesosphere, are still catching the
Sun from below the horizon. That geometry tells you exactly *when* and *where* to
look, and this module turns it into an automatic watch.

## When and where it looks

* **When** — only while the Sun is roughly **6–16° below the horizon**. Brighter
  than that the twilight sky drowns them; darker than that the clouds themselves
  fall into Earth's shadow. The module computes the Sun's altitude with `ephem`
  (falling back to Allsky's `AS_SUN_ALTITUDE`) and skips every frame outside the
  window — so you can safely enable it on both the day and night flows.
* **Where** — low above the horizon, **toward the Sun's below-horizon azimuth**
  (in a Central-European summer: NW→N at dusk, N→NE at dawn). With a fisheye
  `calibration.json` the search band follows the Sun's azimuth exactly; without
  one it falls back to a generic horizon band.

## What counts as a candidate

Inside that band, a NLC is a feature that is all of these at once:

* **blue** — electric-blue / silvery-white (blue channel well above red). This
  rejects the orange twilight glow near the Sun and orange light-pollution domes.
* **brighter than the smooth twilight glow** — structured brightening, not the
  broad twilight arch (removed with a large-kernel background subtraction).
* **structured and extended** — real ripples and bands, not point sources
  (stars are removed with a median filter; only coherent blobs are kept).

The single hardest false positive — the bright rim where the glowing low sky
meets the dark vignette, horizon or treeline — is removed by **eroding the
bright-sky region inward** before looking for structure, so only genuine
interior sky is scored. Validated against a full clear (NLC-free) twilight
sequence: **zero false positives** across the whole arc.

> **Honest scope.** Cleanly separating NLC from twilight-lit thin cirrus is hard
> to fully automate — this is a *candidate* detector. It flags likely nights,
> saves a thumbnail so you can confirm by eye, and pairs with
> [allsky_skyalert](https://github.com/benhartwich/allsky-skyalert) for a
> "possible NLC now — look at the NW" push.

> **Field-of-view caveat.** A strongly zoomed all-sky lens may not image the sky
> right down to the horizon (its fisheye circle is larger than the sensor). NLC
> live low, so with such a lens only the **upper part of a strong display** is
> caught, and only in the azimuths the frame actually reaches. The band mask is
> always clipped to the visible frame, so the index is computed over whatever sky
> is genuinely in view.

## Installation

```bash
cp allsky_nlc.py ~/allsky/scripts/modules/
cp calibration.json ~/allsky/scripts/modules/   # optional but recommended
```

Enable **"Noctilucent Cloud Detector"** in the Allsky WebUI for **both** the day
and night flows. Two things matter for the placement:

1. **Run it *before* `allsky_overlay`** — otherwise it sees the burnt-in compass
   rose / text as bright, "structured" false features.
2. **Set a sky mask** (`meteor_mask.png`) so the treeline and horizon are ignored.

Uses only `cv2` + `numpy` + `ephem`, all already in the Allsky venv.

### calibration.json

The Sun-following band needs the camera's fisheye geometry (centre, radial
distortion, rotation, handedness). This is a fixed, one-time calibration —
`calibration.json` here is the author's camera; generate your own with the
[allsky_meteordetect](https://github.com/benhartwich/allsky-meteordetect)
`tools/calibrate_fisheye.py`. Without it the module still works, using a generic
low-horizon band.

## Configuration

| Setting | Default | Meaning |
|---|---|---|
| Sun altitude — dark limit | `-16` | Below this the clouds are in shadow; stop looking |
| Sun altitude — bright limit | `-9` | Above this the twilight sky is too bright |
| ROI band — bottom / top altitude | `12` / `45` | Search band above the horizon (clipped to what the lens sees) |
| ROI band — azimuth half-width | `75` | Half-width of the band either side of the Sun's azimuth |
| Min blue excess (B−R) | `8` | How blue a candidate pixel must be |
| Min brightness residual | `10` | How far above the smooth glow a feature must sit |
| Bright ceiling | `245` | Reject blown-out sky / burnt-in graphics |
| Edge erosion (px) | `8` | Shrink the bright sky inward to kill vignette/tree edges |
| Detection threshold (% of band) | `0.4` | Flag when this share of the band is blue+bright+structured |
| Sky mask | `meteor_mask.png` | Black = ignore (trees / horizon) |
| History (hours) | `72` | How much history to keep in `nlc.json` |

## Output

- Environment variables `AS_NLC` (0/1), `AS_NLC_INDEX`, `AS_NLC_SUNALT`,
  `AS_NLC_SUNAZ`, `AS_NLC_DIR` — usable in the Allsky overlay.
- A rolling **`nlc.json`**: one `{t, sun_alt, sun_az, dir, index, structure,
  blue, blobs, nlc}` record per in-window frame, ready for a dashboard.
- A colour **thumbnail** of the band (`nlc/nlc-YYYYMMDD_HHMMSS.jpg`) whenever a
  candidate is flagged, for eyeball confirmation.

## Credits

- [Allsky](https://github.com/AllskyTeam/allsky) by Thomas Jacquin and team.
- Built for [astronomy.garden](https://astronomy.garden).

## License

MIT — see [LICENSE](LICENSE).
