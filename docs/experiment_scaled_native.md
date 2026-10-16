# E1 "Scaled-Native": recording guide

Written for: you, at the table with a phone, a tablet and two objects.

**One line.** You record about 20 short clips of a hand putting the real bowl on the real plate. The robot never sees them. They are turned into robot demos that start from the robot's home pose, in **LIBERO's own scene with the bowl and plate rescaled to the real sizes**. A small policy is trained on those robot demos and tested in the same scene.

**Why this is "E1".** Three experiments reuse **one** set of recordings. Only the simulator scene changes:

| | Sim scene | Question it answers |
|---|---|---|
| **E1 Scaled-Native** | LIBERO's table, robot and light. Bowl and plate scaled to the measured real sizes | Does motion from a hand transfer when the *geometry* is right? |
| E2 Scanned-Objects | same, with the bowl and plate drawn as the real objects' splats | Does looking like the real objects help or hurt? |
| E3 Scanned-Scene | the whole scanned table and room, composited | Does a real-looking scene help? |

So you record once. Keep the raw videos and the extracted demos; E2 and E3 regenerate their robot data from the same demos.

Notebook: [`examples/08_scaled_native_experiment.ipynb`](../examples/08_scaled_native_experiment.ipynb) (runs the whole chain on stand-in demos). Design log: D-010 in [design_decisions.md](design_decisions.md).

---

## 1. Before you start

| Do this | Why |
|---|---|
| Leave the **tablet with the ChArUco board** and the **table** where they were for the scans (`data/raw/setup_01`). | The board is the table frame. The scans' splats live in the same frame, so E2 and E3 need no alignment. |
| Measure the **bowl height** and put it in `measurements.yaml` (it is marked `TODO`, now 0.067 m). | It sets how high the fingers close. |
| Use the **same bowl and plate** as in the scans. | E2 attaches their splats. |
| Print `bowl_pad.png` and `plate_marker.png` at **100 %** on A4 landscape, then measure the black square: it must be as set in `RecordingSpec.marker_length_m` (**57.5 mm** measured on the first print: the printer scaled it). | The marker size sets the metric scale of the object poses. |
| Phone: 1x main lens, 1080p30, stabilisation off, focus and exposure locked, no zoom. | One fixed camera model. Same settings as the scans. |

Tablet: white background, night filter off, medium to high brightness, no glare on the screen. Measure one square on the glass with a ruler (measured 37 mm on the first tablet; `square_length_m` in `measurements.yaml`) and do not zoom afterwards.

## 2. Plan the session (once)

```bash
uv run python scripts/16_plan_recording.py --clips 20 --out data/raw/e1
```

This writes into `data/raw/e1/`:

| File | What it is |
|---|---|
| `plan.csv` | One row per clip: where the bowl and plate go (cm from the table's left edge and from your edge), where your hand starts, which side of the rim to pinch |
| `plan_sheet.png` | One small top view per clip. Print it and keep it next to the table |
| `plan.html` | The same, interactive, with the phone drawn in |
| `bowl_pad.png`, `plate_marker.png` | Print these |

It also prints the phone poses that work. The same numbers, for this table and the S24:

**Phone: on the far side, opposite you.**

- Lens about **50 to 70 cm beyond the table's far edge**, roughly in line with the gap between the board and the work area (table x about 0.4 to 0.5 m from the board corner, so about 45 cm from the table's left edge).
- **75 to 95 cm above the tabletop** (about 1.5 to 1.7 m from the floor if the table is 75 cm high).
- Tilted down so the line of sight is about **35 to 45 degrees** below horizontal.
- The board must be **at least 250 px wide** in the frame, the bowl and plate fully inside, and room above them for the hand (30 cm up).

From there the board is large in the frame and the pinch is seen from the front. The search also finds one pose on your side, at the **near-left corner**: lens about 30 cm beyond the near edge, level with the table's left edge, 85 cm up (board 255 px wide, 5 % margin). It is a tighter fit; prefer the far side. Behind your chair in the middle there is no pose that shows the board at 250 px or more; with 1080p video the board there is only about 160 px wide. A 4K recording with intrinsics scaled by 2 would make it work. That has not been tried.

**Does the exact phone position matter?** No. Every clip finds its own camera pose from the board, so a bump between clips is harmless. What matters: the board, both objects and your hand's whole path are in view, and the phone **does not move during a clip**.

## 3. Check before you record the session

Record a **10 s test clip** with everything in place: pad and plate marker down, you do one full pick and place. Then:

```bash
uv run python scripts/17_check_capture.py data/raw/e1/test.mp4 --intrinsics data/camera_intrinsics_s10e.npz
```

It prints PASS, or what to fix: board not found, board too small, a marker not seen in the first second, hand lost, no grasp or release found, grasp not near the bowl. Fix, re-record the test clip, repeat until it passes.

## 4. What goes on the table

```
 far edge  ┌──────────────────────────────────────────────────────┐   phone ▲ (beyond this edge)
           │ [board]                                              │
           │                 ┌ ─ ─ ─ ─ ─ ─ ─ ┐                    │
           │                   plate   bowl pad                   │
           │                 └ ─ ─ ─ ─ ─ ─ ─ ┘  work area          │
 near edge └──────────────────────────────────────────────────────┘
                                   you
```

- **Plate**: marker (id 22) taped flat on the **middle** of the plate's top face.
- **Bowl**: stands in the circle of the **pad** (marker id 21 sits 11.5 cm to the left of the circle). The pad's marker lies on the table, so neither the bowl nor your hand hides it.
- Positions from `plan_sheet.png`. Within about 3 cm is fine: the real position is measured from the markers, not from the plan. The plan only spreads the layouts evenly over the area where the Panda can reach.
- Keep the **board and the markers clear of your hands** for the first second of each clip.
- Do not move the table or the tablet between clips.

## 5. One clip, step by step

Aim for **8 to 12 seconds**. Name the files `clip_001.mp4`, `clip_002.mp4`, and so on, in the order of the plan.

1. Put the bowl pad and the plate at the planned positions. Hands away.
2. Start recording. **Stay out of the frame for one second**: the pad and plate markers are read from the first 0.6 s.
3. Bring your hand in from the **planned start** (the sheet says where: on the table at the left, raised above the middle, and so on). Do not tidy it into a standard start; the variety is the point.
4. **Open the hand wide**, thumb and index tip at least 5 cm apart, while it is in view for half a second or more.
5. Approach the bowl and **pinch the rim on the side the sheet says** (left or right as you see it): thumb on the outside, index on the inside, about 1 cm below the rim edge. Come in from above or from the side, not from far away at table height.
6. **Close the pinch fully** (tips almost touching) and **hold it about half a second**.
7. Lift 5 to 10 cm and carry the bowl to the plate in a smooth arc. Put it down **centred** on the plate marker.
8. **Hold half a second, then open the hand wide** (more than 5 cm) and take it away.
9. Stop recording.

If you fumble (a regrasp, the bowl slips, a second hand is in view), re-record the clip. A bad clip costs more than a missing one.

## 6. Why these rules: how a hand clip becomes a robot demo

This is what the pipeline does with a clip, and why each rule above exists.

| Step | What happens | Rule it explains |
|---|---|---|
| Table frame | The ChArUco board gives the camera pose, once per clip (the phone is static). Everything else is expressed in the table frame, tabletop at z = 0 | Keep the board in view; do not move the phone during a clip |
| Objects | Pad marker plus a fixed offset gives the bowl centre; the plate marker gives the plate centre. Read in the first 0.6 s | Hands off the markers at the start |
| Hand | MediaPipe landmarks, solvePnP for a metric pose. The tip midpoint of thumb and index is the gripper point | Show the whole hand; no gloves |
| Depth | Hand-size error makes the depth wrong by 10 to 20 %. The camera ray through the grasp is intersected with the plane at the pinch height (bowl height minus 1.2 cm), and the whole path is scaled to match | Pinch near the rim top; the pinch height is the one fixed fact the depth correction uses |
| Grasp and release | Thumb-index distance with two thresholds: below 3 cm is closed, above 5 cm is open again | Pinch fully, release wide, one grasp and one release per clip |
| Retarget | Only the **last 20 cm of the approach** is kept, relative to the bowl; the grasp is snapped onto the rim; the carry path is bent to start at the bowl and end centred on the plate | Where your hand started does not matter; how it approaches does |
| Robot | The Panda starts from **home**, goes above the first kept point at a safe height, follows the approach, closes, lifts, carries the *bowl* along the bent path, lowers it, opens, retreats | Robot demos always start at home; only the approach direction and the carry arc come from you |

What is **not** taken from you: the hand's orientation, its speed, where it started, the placement error. The gripper stays top-down and turns about the vertical axis only, so that its fingers close across the rim wall. This is why a left or right pinch works best: the closing axis is then along the table's x axis, which is the robot's default finger axis, and the pinch is seen from the side by the phone.

**Why success needs a centred placement in the sim.** LIBERO's check wants the bowl within 3 cm of the plate's centre, touching it. The retargeter centres the bowl itself, so your placement error does not matter, but a clip where you drop the bowl far off-centre has an odd carry arc and is flagged.

## 7. After recording

```bash
# human clips -> table-frame demos + a QC report (reads sizes from measurements.yaml)
uv run python scripts/02_extract_demos.py data/raw/e1/clip_*.mp4 \
    --setup data/raw/setup_01/measurements.yaml \
    --intrinsics data/camera_intrinsics_s10e.npz --out data/e1/demos
```

It prints `OK` or `FLAG` per clip with the reasons and writes `data/e1/demos/summary.json`. Clips are never dropped silently; a flagged clip is skipped by the next steps.

| Reason printed | Likely cause | Fix |
|---|---|---|
| `table marker never detected` | board not found in any frame | brightness, glare, nearer phone, 4K |
| `bowl marker not seen in the first 0.6s` | pad hidden by a hand, or out of frame | re-record, hands away at the start |
| `hand detected in N% of frames` | hand partly out of frame, sleeves, motion blur | re-record; keep the path inside the frame |
| `hand lost for X s` | occlusion by the bowl at the grasp | pinch the left/right rim, not the near or far one |
| `no grasp detected` / `no release detected` | pinch not closed below 3 cm, or opened below 5 cm | close fully, open wide |
| `anchor scale ... outside [0.6, 1.6]` | wrong intrinsics, wrong pinch height, wrong bowl height in `measurements.yaml` | check them |
| `grasp N m from bowl marker` | pad moved during the clip, or you grasped elsewhere | re-record |

Then replay them on the robot (videos, one per clip) and generate the robot dataset:

```bash
MUJOCO_GL=egl uv run python scripts/03_replay.py data/e1/demos \
    --setup data/raw/setup_01/measurements.yaml --out data/e1/replay
MUJOCO_GL=egl uv run python scripts/04_generate_dataset.py data/e1/demos \
    --setup data/raw/setup_01/measurements.yaml --count 200 --out data/e1/generated
```

`03_replay.py` puts the robot on **your** layout (the sim twin of the clip). `04_generate_dataset.py` draws new random layouts and keeps only rollouts that LIBERO's check accepts; its output has the feature names, image orientation and state formula of LIBERO's LeRobot datasets.

The optional human dataset in LeRobot format (what you recorded, for the record): `scripts/11_export_human_dataset.py` (see the notebook's command list).

## 8. Train and evaluate

**Train** SmolVLA on the generated data (starts from `lerobot/smolvla_base`; the wrapper adds the camera mapping `image -> camera1`, `image2 -> camera2` that the pretrained checkpoints use):

```bash
uv run python scripts/19_train_smolvla.py data/e1/generated --out outputs/e1_smolvla --steps 20000
```

By default only the action expert is trained (`--train-vlm` adds the language model; `lerobot/smolvla_libero` trained all of it). A 20-step smoke run on 30 stand-in episodes works at batch 4 (about 4 steps/s); the time of a full run at batch 32 has not been measured.

**Evaluate** with our own loop, because `lerobot-eval` rebuilds the stock scene on every reset (which would put the small plate and bowl back). `hurobo.policy_rollout` applies the same processing as `lerobot-eval` (image turn, state, tokenizer, normalisation) and stops at the first success. `scripts/18_benchmark_policy.py` runs checkpoints x conditions (`stock`, `clean`, `scaled`) x replan intervals for N episodes each, with the same layouts for every checkpoint, and writes a resumable JSONL, a summary table (success with a 95 % Wilson interval, share of episodes that lifted the bowl) and videos. Open loop (`--open-loop-dataset`) compares predicted action chunks with a demonstration dataset without the simulator. Notebook: [`examples/09_smolvla_benchmark.ipynb`](../examples/09_smolvla_benchmark.ipynb); log: D-011.

```bash
MUJOCO_GL=egl uv run python scripts/18_benchmark_policy.py --checkpoints lerobot hfvla \
    --conditions stock clean scaled --replan 10 --episodes 10 --out eval/pretrained_libero_checkpoints
MUJOCO_GL=egl uv run python scripts/18_benchmark_policy.py \
    --checkpoint-path e1=outputs/e1_smolvla/checkpoints/last/pretrained_model --conditions scaled
```

---

## What has been checked, and what has not

Checked (stand-in demos, repo code; evidence in D-010):

- The bowl and plate in the sim have the measured diameters (within 2 %; a test).
- The table frame, the ChArUco table-frame estimator and the pad offset recover known poses in rendered frames (tests).
- Retargeted rollouts from the home pose put the scaled bowl on the scaled plate: 79 of 80 on new layouts with the final settings (40 of 40 in a scratch loop, 39 of 40 through `scripts/04_generate_dataset.py`), and 29 of 30 before the work area was narrowed. The input there is a made-up hand path, so this is the success rate of the retargeting and execution, not of anything learned.
- The generated dataset loads with LeRobot and has the LIBERO feature layout.

Not checked:

- **No real clip has been run.** MediaPipe's depth error on a real hand, the plane anchor on real grasps, the marker pad on a real bowl, and whether a person's pinch is read as below 3 cm are all untested.
- The bowl's height is a placeholder in `measurements.yaml`.
- The phone poses come from a geometric search with the 1080p intrinsics; nothing has been filmed from them.
- No policy has been trained on this data beyond a 20-step smoke run. The evaluation loop exists (section 8) and has been run only on pretrained checkpoints.
- The scaled plate is 10 % taller than the real one (LIBERO's proportions); a uniform scale matches the diameter, not both.
- LIBERO's stripes and black bowl look nothing like the real objects. That is E2.
