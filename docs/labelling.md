# Labelling rules

**Rules version 1.** Bump `rules_version` in `configs/dataset.toml` if any rule changes meaning, and
relabel or drop the frames labelled under the old rules.

1. **One polygon per resistor body.** One class: `resistor_body`.
2. **Body only.** The polygon runs along the body edge and around the end caps. Leads are never
   included, even where they leave the cap.
3. **Bands are body.** A band whose colour matches the backdrop is still inside the polygon.
4. **Visible part only.** Where one body lies over another, the lower one's polygon stops at the
   upper one's edge. Do not guess hidden parts.
5. **Leads over a body.** A lead crossing a body does not cut the body's polygon.
6. **Bodies cut by the frame edge** are labelled if any of the body is visible.
7. **Tight, not loose.** Follow the visible edge, including its shadowed side. Do not include the
   shadow on the table.
8. **Anything that is not a resistor body** (loose leads, debris, tools) is not labelled.

When fixing an `autolabel` draft, set `label_source` to `manual` for every polygon you change or add.
