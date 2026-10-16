# Scanning real objects and the scene with a phone

Step by step: what to prepare, how to film, and which commands turn the videos into Gaussian-splat `.ply` files
that `scripts/08_splat_composite.py` can attach to the LIBERO sim.

You end up with three kinds of splat:

| Splat | Contains | Scale | How it is made |
|---|---|---|---|
| **Object** (`<name>_<obj>.ply`) | one bowl / plate / cube | metric (from the board) | cut out of the table scan, or scanned alone (arbitrary scale) |
| **Tabletop** (`<name>_table.ply`) | the table surface, objects removed | metric, z-up table frame | the table scan, objects and background cut away |
| **Background** (`<name>_background.ply`) | walls, floor, everything beyond the table | metric, same table frame | the same scan, outside the tabletop cylinder |

All three come from the **same** video and the **same** ChArUco board, so they share one frame and drop into the
sim without extra alignment.

> Status: `--charuco` has now run end to end on two real phone clips with the board shown on a tablet (corner RMS
> 0.5 to 1.1 mm, design log D-009), but only over part of the table. Read the printed ChArUco RMS (step 5) before
> trusting the scale. The single-object
> mode has run on a stock turntable clip and on two real phone clips of a bowl on a wooden floor. Those clips
> are where the capture advice below comes from: see [design_decisions.md](design_decisions.md) (D-001, D-004)
> and `examples/06_scan_parameter_sweep.ipynb`.

---

## 0. Prepare

### Hardware
- Phone (Galaxy S24 main camera, 1x lens). Tripod is optional for the scan; you walk with it.
- **ChArUco board, printed at 100 % scale** (7 x 5 squares, 35 mm squares, DICT_4X4_50). Glue it flat on card or
  a clipboard; a curled board ruins the metric scale. After printing, measure one square with a ruler: it must be
  35 mm.
- Even, diffuse light. Switch off flickering lamps, avoid hard shadows and windows behind the objects.
- Objects with texture. Plain white/black/glossy objects reconstruct badly; put a patterned cloth, newspaper or
  printed sheet under a plain object.
- Optional: a lazy-Susan/turntable for the single-object scan.

### Board on a tablet instead of paper
Possible, but the first attempt failed: in four of six clips not a single marker was detected. What differed in the two
clips that worked: the board filled a few hundred pixels (close, nearly top-down), and it was readable.
- Set the tablet to a **white** background with the colour/night filter off (it showed white squares as cyan), medium to high
  brightness, no reflection of lamps or windows in the screen.
- Keep the board **large in the frame** (film from 30 to 50 cm) and not seen at a grazing angle in most frames.
- Measure one square and one marker **on the screen** with a ruler, never zoom afterwards, pass them as
  `--square-length` and `--marker-length`, and pass the tablet's thickness as `--board-thickness`.
- Test a clip before scanning it: `hurobo.scan.board_visibility` (example in `examples/06_scan_parameter_sweep.ipynb`, part 4).

### Software check (once)
```bash
cd ~/alfredo/workspace/hurobo
uv sync
uv run python -c "import pycolmap, gsplat, plyfile; print('scan deps ok')"
nvcc --version                      # gsplat JIT-compiles against CUDA 12.6
MUJOCO_GL=egl uv run python scripts/00_check_env.py
```

### Print the board
```bash
uv run python scripts/01_calibrate_camera.py generate-board     # -> data/charuco_board.png
```
Print it with "actual size / 100 %", no "fit to page".

### Phone settings (set once, keep for every clip)
| Setting | Value | Why |
|---|---|---|
| Mode | Pro Video | lets you lock exposure and focus |
| Resolution | 1080p/30 or 4K/30 | the pipeline downsizes to 960 px on the long side, so 1080p is enough |
| Lens | 1x main, **no zoom** | one fixed intrinsic matrix for SfM |
| Stabilisation / Super Steady | **OFF** | stabilisation warps frames and breaks the camera model |
| Focus | manual, locked on the object | refocusing changes the field of view |
| Exposure, ISO, white balance | locked | brightness flicker makes the splat blotchy |
| Shutter | 1/120 s or faster if the light allows | motion blur is the #1 reason SfM fails |

---

## 1. Lay out the scene

1. Clear the table. Put the **ChArUco board flat on the table**, near an edge or corner and **outside the
   robot's workspace** (it stays visible in the splat as a checker patch).
2. Do not move the board for the whole session. Its corner is the origin of the table frame:
   x and y run along the board's edges, z is up.
3. Place the bowl, plate and other objects on the table, spread out with a hand's width between them. Measure
   each object's **centre x, y** relative to the board corner (metres, table frame; z up, board origin = the
   corner of the printed board) and its **radius and height**. You need these for the `--object` flags.
   Tip: a quick way is to take one top-down photo with a ruler laid along the board's x axis.
4. Measure the table too: its centre (x, y) in the same frame and roughly its half-width. This gives
   `--table-centre` and `--table-radius`.

---

## 2. Record the table scan (tabletop + objects + background in one video)

This is the main capture. **One lap per clip, about 30 to 40 s.** The pipeline picks 120 keyframes from the
clip; at 30 to 40 s that is one keyframe every 0.25 to 0.35 s. On the bowl clips, keyframes 0.19 s apart
reconstructed fully and held-out PSNR fell by about 10 dB at 0.58 s apart. A 2 to 3 minute clip would give one
keyframe per 1 to 1.5 s with the default `--frames 120`.

1. Start recording with the board and all objects in view.
2. **Orbit slowly** (a full 360 deg circle) around the table at about 40 to 60 cm radius from the table edge,
   keeping the phone at **about 30 to 45 deg downward** so the table fills the lower two thirds of the frame and
   the room (walls, floor, furniture) is visible in the upper third. That upper third is your background.
3. For more coverage, film a **second lap higher** (phone 60 to 80 cm above the table, looking more steeply
   down) and a third low (just above the table, looking across it). Separate clips are the safer choice: the one
   clip that mixed three elevations registered only 70 of 120 frames with sequential matching (the single lap
   registered all 120) and needed the exhaustive retry. It did still reconstruct (27.8 dB held-out against 31.9 dB
   for the single lap), so this is a preference, not a rule; the two clips differ in more than the elevations.
4. Keep moving smoothly, about 1 full orbit per 40 to 60 s. Never rotate on the spot: you need to **translate**
   so the views triangulate.
5. The board must be visible in a good share of the frames (at least 3 views per corner are needed; aim for
   most of the video).
6. Do not move objects, do not let your hands or **your shadow** into the scene, close the door/curtains so the
   light does not change. A moving shadow crosses the floor in many frames of the 3-angle bowl clip; whether it
   explains that clip's lower PSNR was not tested.
7. Keep the distance to the objects roughly constant (the camera height above the floor varied 1.8x in the
   3-angle clip and 1.15x in the single lap). Not isolated as a cause either.

What the pipeline does with it: 120 sharpest frames are picked, COLMAP registers the cameras (lowered SIFT
threshold; exhaustive matching as a retry when under 80 % register), 3DGS is trained (about 7000 iterations),
then the board fixes metric scale and the table frame.

---

## 3. (Optional) Extra background sweep

Only if the room behind the table is badly reconstructed by step 2 (holes, floaters) and you care about its
appearance in the render.

1. **Leave the board where it is**; it is what keeps this scan in the same frame as the table.
2. Record 60 to 90 s: stand at the table's edge, then walk a short arc (1 to 2 m) sideways while pointing the
   phone away from the table at the walls and floor, with the board visible at the start and end of the clip.
3. Run it as its own scan (step 5 with `--name <name>_room`, no `--object`). Because both scans align to the
   same board, the splats overlay in the table frame.

A pure turn of the phone on the spot gives no baseline and SfM fails; walk a little.

---

## 4. (Optional) Scan one object alone

Use this when an object needs more detail than the table scan gives (bowl interior, underside).

1. Put the object on a **textured** surface at the centre of a turntable or table. A plain white backdrop works,
   but give the object texture. A smooth white object on a low-contrast floor is the hard case: it has almost no
   features of its own, and a patterned cloth or newspaper beside it helps SfM more than anything else.
2. Film one slow 360 deg orbit at 40 to 55 deg elevation, 25 to 40 s, at a constant distance so the object keeps
   the same size in the frame. Film other elevations as separate clips. Keep your shadow off the surface.
   A smooth path is not needed; even coverage is.
3. Single-object mode returns an **arbitrary scale** (it uses no board), so you attach it with
   `MATCH_BODY_XY`. Metric objects come from the table scan instead.

---

## 5. Transfer and run

Copy the clips into `data/raw/` (large files are never committed):
```bash
adb pull /sdcard/DCIM/Camera/20261012_101500.mp4 data/raw/table_scan.mp4
adb pull /sdcard/DCIM/Camera/20261012_103000.mp4 data/raw/bowl_alone.mp4
```
(or USB file transfer / any cloud drive; the filenames are only examples).

### 5a. Table scan: tabletop + objects + background
```bash
uv run python scripts/09_scan_object.py data/raw/table_scan.mp4 --name real --charuco \
    --object bowl  0.10  0.05 0.09 0.08 \
    --object plate -0.05 0.12 0.12 0.04 \
    --table-radius 0.5 --table-centre 0.2 0.1
```
- `--object NAME X Y RADIUS HEIGHT`: metres, table frame (board corner = origin). Repeat for each object.
- `--table-radius` / `--table-centre`: the tabletop cylinder. Everything outside it, plus anything more than
  0.6 m above the table or below the surface, goes to the background splat.
- Optional: `--frames 200` for a longer clip, `--iterations 15000` for a sharper splat (slower).

Outputs in `data/splats/`:
```
real_sfm_frame.ply    raw splat in the SfM frame (arbitrary scale, kept for debugging)
real_table.ply        tabletop only, objects removed, metric, z-up   -> attach with METRIC_TOP
real_bowl.ply         bowl cut out, metric                           -> attach with METRIC_BOTTOM
real_plate.ply        plate cut out, metric                          -> attach with METRIC_BOTTOM
real_background.ply   everything outside the tabletop cylinder, metric
```
and `data/scans/real/preview_photo_vs_splat.png` (a few photo | splat pairs).

Without `--table-radius`, `real_table.ply` still holds table **and** background together.

### 5b. Optional background sweep
```bash
uv run python scripts/09_scan_object.py data/raw/room_sweep.mp4 --name real_room --charuco
```

### 5c. Optional single object alone
```bash
uv run python scripts/09_scan_object.py data/raw/bowl_alone.mp4 --name bowl_alone --holdout-every 10
```
Writes (no board, arbitrary scale):
```
bowl_alone_full.ply     everything the splat contains, uncropped
bowl_alone.ply          cropped to the object (floor included)
bowl_alone_object.ply   the object alone: floor removed, upright, base at z = 0 (needs a fitted floor)
```
Attach `bowl_alone_object.ply` with `SplatFit.MATCH_BODY_XY` (its scale is arbitrary), e.g.
`build_assets(bowl_ply=Path("data/splats/bowl_alone_object.ply"))` in `examples/01_splats_and_scans.ipynb`.
and prints how the crop was chosen. In order: `--crop X Y R` if you gave it; the floor-plane + cluster estimate
(an object standing on a floor or table); otherwise the old percentile fit, which assumes the SfM points sit on
the object (a turntable on a plain backdrop). `--holdout-every 10` withholds every 10th keyframe from training
and reports PSNR on them, which is the number to trust; the training-view PSNR is optimistic.

To change only the crop, without re-running SfM or training:
```bash
uv run python scripts/09_scan_object.py data/raw/bowl_alone.mp4 --name bowl_alone --recrop --crop 0.1 0.5 2.7
```
Units are the splat's own (SfM) units; pick the centre and radius with `crop_figure` in
`examples/06_scan_parameter_sweep.ipynb`.

### 5d. Optional collision shape for an object
```bash
uv run python scripts/10_build_collider.py --ply data/splats/real_bowl.ply --name real_bowl
```

---

## 6. Check the result before using it

1. **Read the console.** Expect:
   ```
   [SfM] registered 118/120 images in 1 model(s)
   [scan] ChArUco alignment: scale 0.xxxx m/unit, corner RMS 2.1 mm
   ```
   - `only N/120 registered; retrying with exhaustive matching` is the automatic fallback. It is slower
     (about 6 min for 120 images) but it is what made a harder clip register fully.
   - A warning that SfM registered under 80 % of the keyframes: orbit too fast, blurry, or too little texture.
     Re-film. More than one model means COLMAP split the orbit.
   - Corner RMS above ~5 mm: the board was curled, mis-measured, or seen in too few views. Re-film or re-print.
2. **Open the preview** `data/scans/real/preview_photo_vs_splat.png`. Splat should resemble the photo; fuzzy
   floaters near the camera tracks are normal, large holes are not.
3. **Look at it in 3D**: open `examples/01_splats_and_scans.ipynb` and point it at `data/splats/real_*.ply`.
   Check: the table is flat at z = 0, the bowl sits on it, and the background sits outside the cylinder.
4. **Composite into the sim** (the real test):
   ```bash
   MUJOCO_GL=egl uv run python scripts/08_splat_composite.py --help
   ```
   and check the objects land on the sim's table at the right size.

The PSNR printed at the end is measured on the training views unless you passed `--holdout-every`, so it is
optimistic. PSNR is also only measured on frames that registered: a reconstruction that covers a short arc can
score well. Look at the registered fraction and how much of the orbit is covered
(`examples/06_scan_parameter_sweep.ipynb`) before the PSNR.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SfM registered no images` | too fast, blurry, textureless | slower orbit, more light, add texture under objects |
| warning: registered under 80 % of the keyframes | too few features or weak overlap between frames | shorter clip (keyframes 0.2 to 0.3 s apart), slower lap, texture beside the object; see design log D-001 |
| single-object crop is a wide floor disc with a tiny object | no floor plane found (fell back to the percentile fit), or the cluster is wrong | `--recrop --crop X Y R` with values from `crop_figure` |
| `only N ChArUco corners seen` | board out of view / too small / blurry | keep the board larger in frame, film closer, more light |
| Table not flat in the viewer | the printed board is warped | glue the board to something rigid |
| Wrong size in the sim | printed board not 35 mm squares | re-print at 100 %, measure a square |
| Background full of floaters | few views of the room | do the optional sweep (step 3), or just discard `_background.ply` and keep the sim's own background |
| Object cut includes table | wrong radius/centre | adjust `--object` and re-run; the whole run takes about 3 min on the dummy clip |
| Colours flicker across the splat | auto-exposure/white balance | lock both on the phone |
| `gsplat` compile error | torch/CUDA mismatch | torch must be the cu126 build and `nvcc` 12.6 on PATH (see CLAUDE.md) |
