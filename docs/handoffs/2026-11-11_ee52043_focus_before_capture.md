# Handoff to EE52043: check the camera's focus before every capture (2026-11-11)

From robot_offline's band-reading session to the EE52043 session. **Every capture from the camera must be
taken right after a focus check on the plate's centre cross, done while the arm is parked.** This
replaces the fixed focus (`camera.toml`, 306) for every capture. It also replaces the warm-up step of
`2026-11-11_ee52043_read_once_plan.md`: with focus checked before each capture, a warm-up isn't needed.
**The check is ready to use:** `robot_offline.pre.focus.FocusTracker` (below).

## Why

The camera's focus number doesn't hold the lens in place. The IMX708 focuses with a voice-coil motor,
open loop: the number sets the coil's current, and nothing measures where the lens ends up. Measured on
2026-11-11 (`experiments/2026-11-11_focus_drift` on the band-reading PC):

- **Nothing on the rig moved.** Every plate fit since 2026-11-06 puts the camera in the same place to
  within 3 mm, worth about 2 focus settings.
- **From cold, the best focus climbs from about 299 to about 337 as the camera warms,** with a time
  constant of about 3 minutes: 6.4 settings a minute at 2 minutes, 0.5 at 10 minutes. Released and idle, it
  cools and falls back within minutes.
- **Reopening** the camera shifts it ±2–3. The lens has about **3 settings of hysteresis**, depending on
  the direction it arrives from.
- **About 15 settings off, a body is half as sharp.** On the blurred plate read that started this, the
  band crops were a quarter of their usual sharpness, close-ups included, and the reader's values couldn't
  be trusted.
- The centre cross is a good reference: **one offset measured on it corrects focus for bodies at every
  distance**, to within about ±5.

Segmentation tolerates blur far better than band reading does, but it also misses more small bodies when
they're soft. Every capture should be in focus.

## The rule

**Before each capture, with the arm parked and the plate in view:**
1. **Focus check on the centre cross:**
   - **first park after the camera is opened:** a full search, 270–380 in steps of 5, then the peak
     interpolated. **18.6 s** (measured).
   - **every park after that:** a narrow re-check, ±6 around the last best in steps of 3, with 6 frames to
     settle and 1 frame per step. **3.4 s** (measured). Expected later: about 2 s with ±3, about 1–1.5 s with
     per-frame stepping (not yet tested).
   - **Always approach from below:** set focus about 10 under the first step before starting. That
     removes the hysteresis.
2. **Set the focus to the check's best**, and take the capture: 8 frames averaged, as now.
3. **Close-ups** (if used) at the close-up rule's focus **plus the offset**. The offset is the check's
   best minus the rule's prediction at the cross's distance. Close-ups follow within seconds, so the same
   check covers them.

**Keep the camera open for the whole run.** The check works either way, but a camera kept open drifts
less between parks, so the narrow re-check finds the peak inside its window.

**If the cross is covered** (a part or lead on it; segmentation shows a body there, or the sharpness
curve comes back flat with no clear peak):
- use a body near the centre as the reference: its best focus minus the rule at its distance gives the
  offset;
- if there's nothing usable, keep the last focus and offset, and flag the capture as possibly soft.

**If the best lands at an edge of the narrow window,** the lens moved more than expected: widen the
search (or run the full one) before capturing.

## In robot_offline: `pre/focus.py` (available)

```python
from robot_offline.pre import focus

tracker = focus.FocusTracker(cap, cal)     # once the plate is found; cap is the open camera
...
tracker.cal = cal                          # if the plate fit changed since
result = tracker.check(polygons=None)      # before each capture, arm parked: sets the camera to the best focus
result.focus, result.offset                # the focus set; the lens's offset from the close-up rule
result.reference                           # "cross", "body" (cross covered, a body used) or "kept" (no clear peak)
result.kind, result.seconds                # "full" (first check, or the lens moved past the window) or "narrow"
tracker.status()                           # one line for a display or log
```

- **`check` does everything in the rule:**
  - the first call is the full search, later calls the narrow re-check, widened when the peak lands at an
    edge;
  - every search approaches from below;
  - it ends with the camera at the best focus, ready to capture.
- **Pass `polygons`** (the last segmentation's outlines, rectified px) when you have them: they're the
  fallback reference if the cross is covered.
- **It saves each result to `data/plate/focus.json`** (robot_offline's data folder).
  `closeup.take(...)` reads it while it's under 2 minutes old: each close-up's focus is the rule plus the
  offset, and afterwards the focus goes back to the checked one, not camera.toml's.
- robot_offline's tools all use it: `tools/read_plate.py` and `tools/collect_bands.py` check before every
  capture. `inf/live_detect.py` checks when the plate is found, on `f`, and before close-ups (`u`).
- **Measured on the rig, 2026-11-11** (read_plate, 8 kit resistors, camera just opened):
  - the full check took 16.4 s and set focus 314 (+2.8 on the rule);
  - segmentation found 8 of 8 bodies, against 5 of 8 at the fixed 306 an hour before;
  - band-crop sharpness was 3,200–13,700, against 1,200–4,400 (close-ups 6,500–15,000, against
    1,300–4,400);
  - 7 of 8 bodies read the same value from the main capture and the close-up.
  - Later re-checks take about 3.5 s (measured with the same steps).

Use this API; don't copy the experiment's scripts into EE52043.

## What this changes in the earlier handoffs

- **`2026-11-11_ee52043_band_values.md`** (the reader, `values.read_frame`, the provisional 0.7 limit)
  stands. Add the focus check before the capture that `read_frame` reads.
- **`2026-11-11_ee52043_read_once_plan.md`:**
  - **Its warm-up step is replaced by this rule.**
  - Reading every value once at the start of a run and matching by position each cycle still works, and
    saves the reading time.
  - But with focus checked before every capture, **reading values again at each capture is also sound**,
    and it keeps a moved or newly found body from going without a value.
  - Which one EE52043 uses is its call.

## The cycle, with the check

| Step | Arm | Camera |
|---|---|---|
| park | moves out of view | (idle) |
| **focus check** | parked | full search (first park), else narrow re-check (about 2–3.5 s) |
| capture | parked | 8 frames averaged at the checked focus |
| segment, read values (and close-ups at rule plus offset) | parked | close-ups, about 1.5 s a body, if used |
| pick and place | moves | (idle; keep the camera open) |
