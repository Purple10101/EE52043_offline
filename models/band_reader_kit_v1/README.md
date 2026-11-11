# band_reader_kit_v1: the band reader for the resistor kit (2026-11-11)

Reads a resistor's value from its colour bands: `post/bands/learned.py`, BandCTC (the body read column by
column). It is live_detect's default `--reader`. Load it with `learned.load(path)`, which also sets its
temperature and limits decoding to its values.

| | |
|---|---|
| Trained on | Every kit value collected by 2026-11-11 (31 values from 0R to 1M, all blue 1% 5-band parts; 2,069 crops) plus 12,387 synthetic crops. Started from `runs/band_kit_v2`, 40 epochs, best of 3 seeds (seed 2, epoch 36). No HiRes photos. |
| Decodes | Only the kit's 52 values (configs/resistor_kit.toml; the card's 23k taken as 33k), each as its 4- and 5-band code: 103 codes. A misread has to land on another kit value to come out wrong. |
| Temperature | 2.34, fitted on held-out placements (scores divided by it). |
| Confidence limit | **0.998 to act on a value without measuring.** Training's limit, the lowest with no wrong answer among the 518 held-out placement crops, was 0.99697, scored on the GPU. Scored on the CPU, as the robot runs, one wrong crop (5k1 read as 510R) sits at 0.99697 too, so add a margin. At 0.998: 51% of crops answered, 0 wrong. At 0.99: 72%, 1 wrong. At 0.95: 88%, 3 wrong (1 body). |
| Held-out placements | best guess right 97.5% of crops (one placement per value never trained on); colours read right with no list of values: 94.4% |

**Limits:**
- **Values it has seen:** the held-out placements are new captures of values it trained on. On values it
  had never seen, the same method (v2) got 53–64%: `experiments/2026-11-11_band_reader_kit` §6.
- **Parts from outside the kit:** a light-green part was read as 330k at 0.96–0.98. Decoding only kit
  values doesn't stop that; the robot should only be given kit parts.
- **Live check on 12 single parts (§8 captures):** confident and right on 5 kit parts, plus a beige
  4-band 1k. 4 kit parts were held back (best guesses on 1k and 15R were multiplier mix-ups below the
  limit). No confident wrong answer on a known kit part.
- **Blue bodies only, from one evening's lighting.**

**The tools answer at `learned.MIN_CONFIDENCE` (0.95).** On held-out placements that gives about 0.6%
wrong (3 of 518 crops, 1 of 259 bodies). Where a value is acted on without measuring, use this model's
limit with its margin, 0.998. The limit isn't yet saved with the model or applied automatically.

**Its wrong answers on held-out placements are multiplier mix-ups** (5k1 read as 510R, 1k5 as 150R:
brown against black on a blue body), plus one direction error (120R read as 10k).

Details: `experiments/2026-11-11_band_reader_kit/README.md` §9. Report: `report.json`.
