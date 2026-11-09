# Labelling rules

**Rules version 2** (2026-11-09). Bump `rules_version` in `configs/dataset.toml` if any rule changes
meaning, and relabel or mark the frames labelled under the old rules.

1. **One polygon per object of a pickable size.** One class: `object`. Resistors, diodes, capacitors,
   op-amps and other packages, transistors, screws, nuts and anything else about 2 to 25 mm across lying
   on the plate. What kind it is (`kind` in the manifest: `resistor`, `diode`, ...) is recorded where
   known, but is not the class.
2. **Body only.** The polygon runs along the body's edge. Leads and pins are never included, even
   where they leave the body. An object without leads (a screw, a nut, a lump) is outlined whole.
3. **Markings are body.** A band, stripe or print whose colour matches the backdrop is still inside
   the polygon.
4. **Visible part only.** Where one object lies over another, the lower one's polygon stops at the
   upper one's edge. Do not guess hidden parts.
5. **Leads over a body.** A lead crossing a body does not cut the body's polygon.
6. **Objects cut by the frame edge** are labelled if any of the body is visible.
7. **Tight, not loose.** Follow the visible edge, including the side facing the camera on a tall
   object. Do not include the shadow on the table.
8. **Not objects, never labelled:** loose leads and lead offcuts, crumbs, wood shavings, pencil marks,
   smudges, and tools or anything larger than the pickable size.

**What each frame has labelled** (`labelled` on each manifest sample):
- `"objects"`: every object is outlined, under these rules.
- `"resistors"`: only resistor bodies are outlined (rules version 1). Other objects in the frame
  may be present but unlabelled. Every frame labelled before 2026-11-09 is marked this way.

These frames still score resistors. But they must not be used to train or score the object class,
because an unlabelled screw there would count as background.

When fixing an `autolabel` draft, set `label_source` to `manual` for every polygon you change or add.

## Changes

- **Version 2 (2026-11-09):** the class became `object`, every object of a pickable size rather than
  resistor bodies only (`experiments/2026-11-09_object_seg_design`). Resistor outlines are unchanged.
- **Version 1:** one class, `resistor_body`.
