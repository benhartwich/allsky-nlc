# Changelog

## v0.1.1
- Upload the candidate thumbnail to the remote website (`nlc/` subfolder) so the
  dashboard banner's image link resolves remotely, not just locally.

## v0.1.0
- Initial release.
- Sun-elevation-gated noctilucent-cloud candidate detector: runs on the day and
  night flows, active only while the Sun is 6–16° below the horizon.
- Search band follows the Sun's below-horizon azimuth low on the horizon, using a
  fisheye `calibration.json` (with a generic horizon-band fallback).
- Candidate signature: blue excess (B−R) + brightness above the smooth twilight
  glow + coherent, extended structure; stars removed with a median filter.
- Interior-erosion of the bright-sky region removes the vignette-rim / horizon /
  treeline false "structure"; saturation ceiling rejects blown sky and burnt-in
  overlay graphics. Validated at zero false positives over a full clear twilight
  sequence.
- Rolling `nlc.json` for charting, candidate thumbnails for confirmation, and
  overlay environment variables. Pairs with allsky_skyalert for push alerts.
