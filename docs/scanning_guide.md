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

> Status: `--charuco` has been verified on synthetic board views and the single-object mode on a stock turntable
> clip (see `PLAN.md` §5). The `--charuco` path has **not** been run end to end on a real phone video yet, so
> read the printed ChArUco RMS (step 5) before trusting the scale.

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

This is the main capture. 2 to 3 minutes.

1. Start recording with the board and all objects in view.
2. **Orbit slowly** (a full 360 deg circle) around the table at about 40 to 60 cm radius from the table edge,
   keeping the phone at **about 30 to 45 deg downward** so the table fills the lower two thirds of the frame and
   the room (walls, floor, furniture) is visible in the upper third. That upper third is your background.
3. Do a **second lap higher** (phone 60 to 80 cm above the table, looking more steeply down) and, if you can, a
   third lap low (just above the table, looking across it) so object sides and rims are seen.
4. Keep moving smoothly, about 1 full orbit per 40 to 60 s. Never rotate on the spot: you need to **translate**
   so the views triangulate.
5. The board must be visible in a good share of the frames (at least 3 views per corner are needed; aim for
   most of the video).
6. Do not move objects, do not let your hands or shadow into the scene, close the door/curtains so the light does
   not change.

What the pipeline does with it: 120 sharpest frames are picked, COLMAP registers the cameras, 3DGS is trained
(about 7000 iterations), then the board fixes metric scale and the table frame.

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
   but give the object texture.
2. Film one slow 360 deg orbit at about 40 deg elevation, 40 s, then one lower and one higher lap if you can.
   The object must stay at roughly the same size in the frame.
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
uv run python scripts/09_scan_object.py data/raw/bowl_alone.mp4 --name bowl_alone
```
Writes `data/splats/bowl_alone.ply` (no board, arbitrary scale, cropped to the object).

### 5d. Optional collision shape for an object
```bash
uv run python scripts/10_build_collider.py --ply data/splats/real_bowl.ply --name real_bowl
```

---

## 6. Check the result before using it

1. **Read the console.** Expect:
   ```
   [SfM] registered 118/120 images
   [scan] ChArUco alignment: scale 0.xxxx m/unit, corner RMS 2.1 mm
   ```
   - Registered fewer than ~80 % of the images: orbit too fast, blurry, or too little texture. Re-film.
   - Corner RMS above ~5 mm: the board was curled, mis-measured, or seen in too few views. Re-film or re-print.
2. **Open the preview** `data/scans/real/preview_photo_vs_splat.png`. Splat should resemble the photo; fuzzy
   floaters near the camera tracks are normal, large holes are not.
3. **Look at it in 3D**: open `examples/01_splat_viewer_3d.ipynb` and point it at `data/splats/real_*.ply`.
   Check: the table is flat at z = 0, the bowl sits on it, and the background sits outside the cylinder.
4. **Composite into the sim** (the real test):
   ```bash
   MUJOCO_GL=egl uv run python scripts/08_splat_composite.py --help
   ```
   and check the objects land on the sim's table at the right size.

The PSNR printed at the end is measured on the training views, so it is optimistic. Report it as such.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SfM registered no images` | too fast, blurry, textureless | slower orbit, more light, add texture under objects |
| `only N ChArUco corners seen` | board out of view / too small / blurry | keep the board larger in frame, film closer, more light |
| Table not flat in the viewer | the printed board is warped | glue the board to something rigid |
| Wrong size in the sim | printed board not 35 mm squares | re-print at 100 %, measure a square |
| Background full of floaters | few views of the room | do the optional sweep (step 3), or just discard `_background.ply` and keep the sim's own background |
| Object cut includes table | wrong radius/centre | adjust `--object` and re-run; the whole run takes about 3 min on the dummy clip |
| Colours flicker across the splat | auto-exposure/white balance | lock both on the phone |
| `gsplat` compile error | torch/CUDA mismatch | torch must be the cu126 build and `nvcc` 12.6 on PATH (see CLAUDE.md) |
