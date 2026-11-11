# Handoff to EE52043: read the band values once, at the start of a sorting run (2026-11-11)

> **Updated later on 2026-11-11:** focus is now checked before **every** capture
> (`2026-11-11_ee52043_focus_before_capture.md`). That replaces this note's warm-up step and its interim
> calibration. Reading once at the start still works, but reading at each capture is now sound too.

From robot_offline's band-reading session to the EE52043 session. This follows
`2026-11-11_ee52043_band_values.md` (the reader, the `values.read_frame` call, the provisional 0.7 limit)
and changes **when** values are read: once, at the start of each sorting run, not every cycle. It's the
plan the user and band reading agreed on 2026-11-11. The EE52043 changes are yours to design.

## Why: the camera's focus drifts while it runs

The band reader needs sharp crops. On 2026-11-11 a plate read came out at a quarter of its usual sharpness,
close-ups included, with nothing moved (`experiments/2026-11-11_focus_drift`, on the band-reading PC):

- The camera's focus number doesn't fix where the lens is. The IMX708 focuses with a voice-coil motor,
  open loop: the number sets the coil's current. Every plate fit since 2026-11-06 puts the camera itself
  in the same place to within 3 mm.
- **While the camera is in use, the best focus climbs:** about 4 settings a minute, +17 in 4 minutes, not
  levelling off by then. It falls back when the camera sits idle. Reopening it shifts it ±2–3, and the lens
  shows about 3 settings of hysteresis.
- About 15 settings off, the bands are half as sharp. So `camera.toml`'s fixed focus (306), and the
  close-up rule fitted on 2026-11-09, are wrong within minutes.
- The plate's **centre cross** works as a focus reference: one offset measured on it corrects the close-up
  rule for bodies at every distance, to within about ±5.

Re-checking focus every cycle would keep up, but it costs seconds each cycle. **Reading every value once at
the start, right after one focus calibration, means calibrating once per run.** Segmentation still runs
each cycle and is far less sensitive to focus (bodies were found at every focus from 298 to 330).

## The plan

### 1. At the start of a sorting run (arm parked, plate loaded)

0. **Warm-up:** the camera open and streaming for about 10–15 minutes since power-on (see "Focus
   calibration"), and kept open from here to the end of the run.
1. **Plate:** `plate_search` as now, with the exposure at the reading level
   (`level=learned.READING_LEVEL`: plate grey 205–225).
2. **Focus:** calibrate once, now (see "Focus calibration" below; interim steps until robot_offline's
   focus check exists).
3. **Capture at the normal frame count:** the mean of 8 frames (`closeup.mean_frame(cap, 8)`), as
   `tools/read_plate.py` and the collector do. The reader was trained and scored on such captures.
4. **Segment** (`seg_grey_v2`, conf 0.5), then **read every body:**
   - `values.read_frame(net, frame, cal, polygons)`: one reading per body from the averaged capture;
   - and a **close-up** per body (`closeup.take(cap, cal, polygons, seg=..., grey=..., camera=...)`), each
     at its own focus, about 1.5 s a body. Read each with
     `learned.log_probs(net, k.crop.image, k.white)` and average it with the main capture's, half and half,
     as `learned.Views.pin` does and as the reader was scored. With 20 bodies the close-ups take about 30 s;
     the lens drifts only about 2 settings in that time, so one calibration covers them.
5. **Keep a table for the run:** per body, an id, its centre and yaw (robot frame, from `Detection`), its
   outline, and its reading (value or None, best guess, confidence, bands). Plan the whole run from it: the
   order of picks, and which parts are already known to need measuring.

### 2. Each cycle after that (no band reading)

1. Segment as now, for positions.
2. **Match each body to the table by position:**
   - centre within about 3 mm and yaw within about 15° of a remembered body keeps that body's value;
   - nearest match first, one to one.
3. Pick and place by the remembered value.

### 3. When a body can't be matched

Picking one part can nudge its neighbours, especially with long, tangled leads. Segmentation also misses
about 1 body in 10 per frame on the kit's small blue parts, mostly on the near side. So in any cycle a body
can turn out to:
- **have moved** beyond the tolerance;
- **be new**: it was missed at the start, or it was under another part;
- **be missing**: it was picked, or it's missed this cycle (keep it in the table for a few cycles before
  dropping it).

**A moved or new body has no trusted value.** Either read it again (a focus check first, then a capture
and close-up of that body: the same as the start-of-run read, for one body), or set it aside: a
"measure later" bin, or leave it on the plate. Never give it a neighbour's value.

## Focus calibration

**Planned in robot_offline** (band reading builds it; not yet available): `pre/focus.py`.
- A search on the centre cross from below. Its best setting becomes the main capture's focus, and its
  offset from the rule shifts every close-up.
- Saved to `data/plate/focus.json`, with a fallback to a body near the centre when the cross is covered.
- About 2–3 s once tuned (now about 6 s).

**Until then, before each run:**
- With the run's parts on the plate (bodies at a spread of distances), run
  `python -m robot_offline.post.bands.closeup calibrate --save --fresh`. It sweeps focus on the bodies and
  refits the close-up rule in `configs/camera.toml [closeup]` (robot_offline's config; band reading owns
  that section). About 1–2 minutes.
- The close-ups are then sharp. The main capture stays at the fixed 306, so **trust the close-up's
  reading** over the main capture's if the two disagree.
- Start the run straight after: the rule goes stale within minutes.

**Warm up first (measured 2026-11-11, cold start).** From cold the best focus climbs from about 299 and
levels off near 337, with a time constant of about 3 minutes:

| Since power-on | 2 min | 5 min | 10 min | 15 min |
|---|---|---|---|---|
| Drift per minute | 6.4 | 2.4 | 0.5 | 0.1 |

So:
- **open the camera and keep it streaming for about 10–15 minutes before the start-of-run calibration and
  read.** After that, one calibration holds for the run;
- **keep the camera open for the whole run.** Released and idle, it cools and drifts back within minutes.

Two caveats are still being checked:
- the end point (about 337) is extrapolated from 6.7 minutes of data;
- those runs swept the focus non-stop. Plain streaming may warm the module less, in which case the warm-up
  should include some focus sweeps.

Band reading will update this note.

## What stays the same (see the first handoff)

- **The reader:** `models/band_reader_kit_v1`, kit values only. Provisional limit 0.7 in
  `configs/band_reader.toml [deploy]`: about 1 body in 80 wrong at 0.7, nearly always a factor of ten.
- **No ohmmeter yet:** unsure parts are set aside, never put in a value bin. Values outside 10R–10k go to
  the reject bin.
- **Kit parts only:** a non-blue or non-kit resistor can be read confidently and wrongly.
- **robot_offline owns** the reader, plate search, segmentation and the focus check. Use them through
  their API, and report problems to the robot_offline sessions rather than patching robot_offline.

## Summary for the cycle

| When | Focus | Band reading | Segmentation |
|---|---|---|---|
| Power-on | warm up 10–15 min, camera kept open | none | none |
| Start of run | calibrate once | every body: 8-frame capture plus close-up, averaged | yes |
| Each cycle | none | none (values from the table) | yes, to match positions |
| A moved, new or unmatched body | check first, if re-reading | that body only, or set it aside | yes |
