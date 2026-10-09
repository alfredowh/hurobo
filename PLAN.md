# ego2robo: Challenge Plan (v3, after the real2sim review)

> **Pitch:** *The robot was never teleoperated. The sim table is a copy of my real one, and the Panda replays what my hand did on it.*
> 15 human demos, recorded third-person on a tripod → a sim digital twin of the real layout and camera → object-centric retargeting → hundreds of verified robot demos in LIBERO → SmolVLA fine-tuned **only** on that data.

Deadline: **Fri 9 Oct 2026, 23:59 BST = Sat 00:59 CEST**. Plan v2 written Thu 8 Oct ~17:00. v3 written Fri 9 Oct ~01:00 CEST after `REAL2SIM_IDEAS.md`'s research pass — **no code existed yet at that point**, so §5's schedule is re-timed from here. See `REAL2SIM_IDEAS.md` for the full writeup (prior work, hardware verdicts on the S10e/CMF watch, frame-convention notes for the twin camera).

---

## 0. Operating mode

See `CLAUDE.md` for standing instructions: Claude's supervisor/developer roles, coding standards, and environment rules. This file covers only the implementation plan and schedule.

**What's new in v3 (see `REAL2SIM_IDEAS.md` for the full reasoning):**
- **Third-person tripod camera, confirmed** — not a late change, v2 already assumed a static phone; v3 just renames "ego video" to "phone video" so the README doesn't contradict the setup.
- **Scene twin added.** On top of v2's object-centric retargeting (§4), a new `SceneTwin` component copies the real camera pose and object layout into the sim and renders from the same viewpoint, so the MVP deliverable becomes a *real ↔ sim-twin* side-by-side rather than a generic LIBERO replay. This is the headline visual. Cited prior work: Video2Sim2Real (arXiv 2606.08828), Phantom (arXiv 2503.00779, visualisation-only reuse — we don't train on edited real frames).
- **S10e and CMF Watch Pro 2, researched and declined for the core path.** Raw phone IMU drifts too fast for position; the watch has no sensor/GPS export worth using. The S10e's one conditional role is a second tripod view for stereo triangulation, gated on the monocular depth gate below actually failing.
- **Schedule re-timed** from "now" (Fri 01:00 CEST) instead of from the original Thu 17:00 anchor, since setup hadn't started.

---

## 1. What changed from v1 (and why)

| # | v1 problem | v2 fix |
|---|---|---|
| 1 | **Depth bug.** Projecting the hand onto the table plane is only correct when the hand touches the table. A hand 10 cm in the air ends up 5 to 15 cm off on the table plane (parallax). | Get metric 3D hand position by running **`solvePnP` between MediaPipe's metric *world* landmarks and its 2D image landmarks**. That gives hand translation in the camera frame on CPU, with no depth sensor. Then **anchor-correct** it: at the grasp frame, the gripper point must coincide with the bowl marker. |
| 2 | **Custom robosuite scene.** It means hand-building assets, and SmolVLA's LIBERO tooling can't be used. | **Mirror an existing LIBERO task in real life** (e.g. LIBERO-Goal *"put the bowl on the plate"* with a real bowl and plate). No sim asset work. `lerobot-eval` works out of the box. Directly comparable to the reference example in the brief. |
| 3 | **BC on 15 replayed demos memorises.** With randomised start positions, success would be near zero. | **Object-centric demo generation (MimicGen-style).** Split each human demo into object-relative segments, re-target them to many sim object poses, execute, and **keep only successes**. 15 human demos become 200–500 verified robot demos. This is the creative core. |
| 4 | **Confounded claim.** Fine-tuning a LIBERO-trained checkpoint proves nothing, because it already knows the task. | Fine-tune from **`lerobot/smolvla_base`**, which never saw LIBERO, **on ego-derived data only**. Any success in sim is then attributable to your data. |
| 5 | **No action-space plan.** | LIBERO uses **7-D relative EEF delta actions in [-1, 1]** (6 pose + 1 gripper) at 20 Hz. A closed-loop `WaypointFollower` converts waypoints into clipped deltas from the *current* sim EEF pose, so it is robust to tracking error. |
| 6 | **Train/eval mismatch risk.** | Generate data **through the same LIBERO env wrapper `lerobot-eval` uses**, so image flips, state layout (8-D: eef pos, axis-angle, 2× gripper qpos), and camera keys are identical by construction. |
| 7 | **No overnight compute plan.** | Demo generation runs overnight on CPU (24 cores). SmolVLA trains Friday morning on the GPU while you write the README. |

---

## 1a. Storyline: what actually has to match between the real objects and the sim, and why

The question to interrogate before trusting §4's method: *if a model is trained from human video and then run on a robot in sim, don't the real and sim objects need the same shape, size, maybe texture?* For a naive "watch the video, imitate the pixels" storyline, yes — and that's disqualifying on a 30-hour budget; there is no real2sim visual-domain-transfer model for LIBERO assets, and training one from 15 clips tonight isn't realistic. The plan sidesteps that requirement entirely, by choice of storyline, not by solving the domain gap. Three storylines considered:

| # | Storyline | What crosses the real→sim boundary | What has to match (real vs. sim) | Verdict |
|---|---|---|---|---|
| A | **Motion-only retargeting (chosen, §4).** The human video is a mocap device, not training data. ArUco + MediaPipe + `solvePnP` extract a TCP trajectory and a grasp signal; these are expressed **object-relatively** and replayed against the *sim's own* object poses. SmolVLA trains only on sim-rendered frames paired with sim actions — it never sees a real pixel. | A handful of numbers per segment: approach direction, grasp height/aperture, lift height, carry-path shape, place offset, release timing. | Only the *task topology*: a graspable concave container and a flat target surface, roughly bowl/plate-scaled, so the extracted grasp-height and aperture numbers are physically sensible on the sim object too. Exact shape, color, texture, material: **irrelevant**, because nothing downstream ever looks at the real object. | **Chosen.** Feasible tonight; the domain gap is avoided, not closed. |
| B | **Pixel-level domain transfer.** Record human video, map real frames into something a sim-trained or jointly-trained policy can consume directly (CycleGAN/diffusion real2sim translation, or a shared visual encoder trained across both domains). | Real RGB pixels, translated or encoded. | Real object shape and scale would need to be close to the sim asset for the translation to generalize at all, and camera intrinsics/extrinsics/lighting start to matter too. | **Rejected.** No existing real2sim model for LIBERO assets; building and validating one from 15 clips in under a day is not a realistic bet. Note as future work in the README. |
| C | **World model / video-prediction planning.** Train or reuse an action-conditioned video predictor on human + sim video, plan robot actions via MPC against predicted future frames. | Latent dynamics, still visually grounded in both domains. | Same visual-similarity requirement as B, plus a working planner/MPC loop on top of the predictor. | **Rejected**, consistent with §2's model-selection pass: no mature local inference path for this at VLA scale, and LIBERO already hands us ground-truth dynamics for free, so predicting them and planning over the prediction buys nothing here. |

**So: no, the bowl and plate on your table don't need to match the sim assets' shape or texture.** What has to hold is that the *motion* you demonstrate — how you approach a graspable concave object, how high you lift it, how you align it over a flat target, when you let go — is a reasonable prior for the *sim's* bowl and plate. That's a claim about **motion transfer across object instances**, not visual transfer across domains, and §4 step 6 already states the simplification this implies (fixed top-down orientation, yaw taken from the object). The README's "what the human data contributes vs. what the sim contributes" split (end of §4) is where this gets argued honestly to a judge — human data supplies strategy, sim supplies exact geometry and appearance.

---

## 2. Base components: which repos, which setup

### Components

| Component | Choice | Why |
|---|---|---|
| Sim + benchmark | **LIBERO** via `hf-libero` (PyPI, pulled in by `lerobot[libero]`). Underneath: robosuite 1.4.0 + MuJoCo. | Same sim as the brief's example. Pip-installable. Linux only, which suits you. |
| Policy framework | **LeRobot** `huggingface/lerobot` ≥ 0.6.1 (latest release v0.6.1, Aug 2026) | `lerobot-train` / `lerobot-eval`, the LIBERO env wrapper, the LeRobotDataset v3 format, SmolVLA, ACT |
| VLA | **SmolVLA**, from `lerobot/smolvla_base` (450M) | Lightweight. Fits a 16 GB GPU. Matches the brief. |
| Fallback policy | **ACT** (`--policy.type=act`, trained from scratch) | Trains in about 1 hour and is strong on single-task LIBERO. Use it if SmolVLA training stalls. |
| Hand tracking | **MediaPipe Tasks `HandLandmarker`** (`hand_landmarker.task` model file) | CPU-only. Gives 2D landmarks plus **metric world landmarks**, which the PnP trick needs. Use the Tasks API; the legacy `mp.solutions` is deprecated or removed. |
| Frames / objects | **OpenCV ArUco** (`cv2.aruco`, DICT_4X4_50): one table marker (world frame) and small markers taped to the bowl and plate | Zero-ML 6-DoF object poses. Bullet-proof. The VLA never sees real pixels, so the markers don't matter visually. |
| Camera intrinsics | **ChArUco board** shown fullscreen on your laptop, 20 frames, `cv2.aruco.calibrateCameraCharuco` | 10 minutes. You need real intrinsics for PnP. |
| Reference checkpoint (optional) | `HuggingFaceVLA/smolvla_libero` **(could not verify the name from this sandbox; check with `hf` CLI)** | Only for an "upper bound" row in the results table. Never used as the init. |

Read-only reference repos (clone into `~/ref/`, never vendor them):
- `github.com/huggingface/lerobot`: read `src/lerobot/envs/libero.py` first (30 min, worth it). It is the exact obs/action convention you must match.
- `github.com/Lifelong-Robot-Learning/LIBERO`: BDDL task files, task names and indices, object names in the sim.
- MimicGen (NVIDIA, 2023) is a concept reference only. Cite it in the README; don't install it.

### Setup-decision research check (supervisor pass, 9 Oct morning)

Before committing further, checked each of the three big choices against current prior work. Verdict for all three: **reuse as-is** — no plan change, documented here so the README's "why" section isn't invented after the fact.

**Sim env — LIBERO, confirmed.** Alternatives checked: ManiSkill (GPU-parallelized, SAPIEN assets — different asset/action conventions, no direct match to the brief's Panda example, and porting would burn hours with no payoff); Meta-World (simpler obs, but its tasks don't map to the brief's bowl/plate pick-place framing); plain robosuite (already the listed fallback if LIBERO install fails, §2 gate). LIBERO stays primary: pip-installable via `hf-libero`, matches the brief's reference example, and `lerobot-eval`/the dataset format are built around it — switching now would also break the "same wrapper for generation and eval" guarantee in §1 fix 6.

**Hand tracking — MediaPipe + ArUco + solvePnP, confirmed; HaWoR considered and rejected for this submission.** HaWoR (Zhang et al., CVPR 2025, arXiv:2501.02973) reconstructs metric-scale hand motion in *world* coordinates from a moving egocentric camera, via an adaptive SLAM framework plus a motion-infiller network plus a "foundation metric model" for scale alignment — three nontrivial components, no pip package, no documented CPU path or inference latency, and the GitHub repo (`ThunderVVV/HaWoR`) is a research codebase, not a maintained library. Two reasons it's the wrong tool here, not just a time-budget call:
1. HaWoR's hard problem is recovering the *camera's own trajectory* (SLAM) so hand motion can be expressed in a world frame when the camera itself moves freely. Our setup fixes the phone on a tripod and pins the world frame with an ArUco table marker instead — so the problem HaWoR solves doesn't exist in our recording setup.
2. ArUco + `solvePnP` against MediaPipe's metric world landmarks already gives metric-scale `T_cam_hand` directly, with the anchor-correction step (§4.4) handling residual depth error. Adopting HaWoR fresh today would mean debugging an unfamiliar SLAM stack on the capture-critical path for an accuracy gain we don't need.
Cite HaWoR in the README as related work for monocular world-space hand reconstruction; don't depend on it.

**Model selection — SmolVLA primary / ACT fallback, confirmed; no world-model route.** Checked against: Octo (93M params, Berkeley — faster and older, but not integrated into LeRobot's current trainer/LIBERO wrapper the way SmolVLA is, so switching costs setup time for a model with weaker reported LIBERO numbers); TinyVLA (0.8–1.4B, CVPR-adjacent 2024/2025 work, strong speed/accuracy trade-off but its own repo and checkpoint format — not a drop-in with `lerobot-train`); OpenVLA / OpenVLA-OFT (7B — OFT's tuned-decoding variant hits strong LIBERO numbers, but at 7B it needs real finetuning wall-clock and VRAM headroom we'd rather spend on iterating the dataset generator); π0 (3B, Physical Intelligence — LeRobot supports it, but 3B finetuning in the remaining hours is a worse bet than SmolVLA's 450M on an A5500 16GB). On "run a big VLA/WM locally via some mechanism": the 2026 toolchain for that (GGUF/AWQ/EXL2 quantization, CPU-offload via llama.cpp-style layer splitting) is built for **text LLMs**, not for VLA action experts or world models — there's no mature quantized-inference path for π0/OpenVLA-scale action policies today, so this would mean building novel infra rather than using one. World models for manipulation (e.g. iVideoGPT, NeurIPS 2024 — action-conditioned video prediction pretrained on ~1.5M robot/human trajectories) are a different research direction: a video/dynamics predictor needs a planner or MPC loop on top to produce actions, which is out of scope when LIBERO already supplies ground-truth dynamics for free. SmolVLA (450M, fits the A5500 natively in bf16, no quantization needed) stays the only VLA in the plan; ACT remains the fallback per the existing risk register.

### Machine (checked)
Ubuntu 22.04 · RTX A5500 Laptop **16 GB** · driver **560.35** · 24 cores · 62 GB RAM · uv 0.11.7 · system ffmpeg 4.4.2 · 97 GB free disk.

### Setup steps (target: 60 min, gate at 18:00)

Run the setup in `CLAUDE.md` (`uv init`, `pyproject.toml`, `uv sync`, env checks).

**Gate (18:00):** torch sees the GPU, `cv2.aruco` imports, and LIBERO renders a frame of the target task. **Fallback:** if the LIBERO env still fails at 18:30, install `robosuite` alone and use its `PickPlace` env for the replay MVP. Drop the VLA comparison and pivot the story to the retargeting and generation quality.

---

## 3. Repo layout

```
ego2robo/
  pyproject.toml  uv.lock  .env.example  README.md  PLAN.md  REAL2SIM_IDEAS.md
  src/ego2robo/
    camera.py        # CameraIntrinsics (dataclass), ChArUco calibration
    markers.py       # MarkerPoseEstimator: per-frame T_cam_table, T_table_bowl, T_table_plate
    hand.py          # HandTcpTracker: MediaPipe → metric TCP (thumb/index-tip midpoint) + pinch aperture
    demo.py          # HumanDemo dataclass + save/load (.npz); GraspSegmenter (grasp/release from aperture w/ hysteresis)
    twin.py          # SceneTwin: real T_table_cam + T_table_bowl/plate → sets sim object qpos + a non-policy MuJoCo camera at the same pose/FOV
    retarget.py      # ObjectCentricRetargeter: human segments + sim object poses → robot waypoints
    sim.py           # LiberoTask: wraps the lerobot LIBERO env; object poses, success check, render
    control.py       # WaypointFollower: waypoints → clipped 7-D relative actions
    generate.py      # DemoGenerator: sample init states, retarget, roll out, keep successes, write LeRobotDataset
    utils/
      transforms.py  # SE(3) helpers, frame conversions (scipy Rotation); cv-to-mujoco camera convention lives here
      signal.py      # smoothing, resampling to 20 Hz, time-scaling
      video.py       # read/write video, side-by-side compositor, robot-overlay alpha composite (stretch)
      plotting.py    # 3D trajectory plots, results figures
  scripts/
    00_check_env.py  01_calibrate_camera.py  02_extract_demos.py
    03_replay.py     04_generate_dataset.py  05_train.sh  06_eval.sh  07_make_figures.py
    # 03_replay.py --twin renders the real-layout/real-viewpoint twin side-by-side (the MVP deliverable, §4/§5)
  data/              # gitignored: raw/, demos/, generated/
  assets/            # small committed media for the README (gifs, plots)
```

---

## 4. Method (the part judges read)

**Real side (perception, CPU only):**
1. Per frame: ArUco gives `T_cam_table` (the table marker is the world frame). Object markers give `T_table_bowl` and `T_table_plate`.
2. MediaPipe gives 21 image landmarks plus 21 metric world landmarks. `solvePnP` gives `T_cam_hand`. The **TCP is the midpoint of the thumb tip and index tip**, which maps naturally to a parallel-jaw gripper. The **gripper command** comes from pinch aperture with hysteresis.
3. Express everything in the table frame. Smooth (Savitzky–Golay) and resample to 20 Hz.
4. **Anchor correction:** at the detected grasp frame, the TCP should equal the bowl's grasp point. Apply the per-episode depth-scale correction that makes it so.
5. Segment into: approach (relative to the bowl) → carry → place (relative to the plate) → release/retreat.

**Scene twin (the MVP deliverable, new in v3 — see `REAL2SIM_IDEAS.md` §2 for the full writeup):**
5a. `SceneTwin` takes the real `T_table_cam`, intrinsics, `T_table_bowl`, `T_table_plate` from one clip and (i) sets the sim bowl/plate free-joint qpos to the real layout (origin = the table marker, mapped to a fixed point on the sim table; clamp and log, never silently rescale, if a real layout falls outside the Panda's reach) and (ii) points a second, **non-policy** MuJoCo camera at `T_world_cam` with `cam_fovy` from `fy`, writing `cam_pos`/`cam_quat`/`cam_fovy` at runtime — the policy's own `agentview` camera is untouched.
5b. Replay that clip's own retargeted waypoints (not a random LIBERO init state) against this twin and render from the twin camera. Composited side-by-side with the real clip at the same angle, this is the headline visual and **the actual MVP** (supersedes the plain "replay in a LIBERO init state" MVP from v2's §5).
5c. **Frame convention trap:** OpenCV camera convention looks down +z with y down; MuJoCo looks down −z with y up, so `R_mj = R_cv · diag(1, −1, −1)`. Verify against a known marker pose before trusting the rendered view.
5d. **Stretch, gated:** alpha-composite a MuJoCo segmentation render of the robot onto the real frame (50% opacity, hand left visible, no inpainting) as a reprojection sanity check of the whole calibration chain. Cut first if time is short. (A separate, more ambitious visual stretch — rendering the *sim* with Gaussian splats of the real objects instead of this overlay direction — is gated into §5's schedule and detailed in `REAL2SIM_IDEAS.md` §7.)

**Sim side (retargeting and generation, unchanged from v2):**
6. For each LIBERO init state, read the sim bowl and plate poses. Transform each human segment from the source-object frame into the sim object's frame (`T_world_bowl_sim · T_bowl_tcp_human`). Interpolate the carry. Orientation stays fixed top-down, with yaw from the object. This simplification is stated openly.
7. `WaypointFollower` steps the env at 20 Hz. The human timing is slowed about 1.5–2× for robot dynamics.
8. **Keep only rollouts where LIBERO reports success.** The generation success rate is itself a metric of retargeting quality.
9. Record obs, actions, and the task string through the same wrapper `lerobot-eval` uses, into a LeRobotDataset v3.

**What the human data contributes:** which object, approach direction, grasp timing, lift height, carry-path shape, place offset relative to the target, and release timing. **What the sim contributes:** the exact object poses. Present this split honestly in the README.

---

## 5. Schedule

**Superseded.** Plan v2's Thursday/Friday schedule below assumed setup started Thu 17:00. No code existed until Fri ~01:00 CEST, so the schedule is re-timed from there (times in CEST; hard deadline Sat 00:59 CEST = Fri 23:59 BST). Order of cuts if running late: photoreal splat stretch (below the table) → robot overlay (§4 step 5d) → SmolVLA (keep ACT) → data-multiplier plot → stereo second view. **Never cut** the twin side-by-side (§4 step 5b) or the README.

| Time | Block | Gate / fallback |
|---|---|---|
| 01:00–02:00 | **Setup** (§2): `uv init`, `pyproject.toml`, `uv sync`, env checks, `00_check_env.py`. Print the table marker (≥ 10 cm), two object markers, generate the ChArUco board. **Then sleep.** | LIBERO renders a frame → otherwise plain robosuite |
| 02:00–08:00 | **Sleep.** | |
| 08:00–09:30 | **Capture.** ChArUco calibration. Record **15 demos**: bowl/plate layouts × reps, third-person tripod, 1080p30, stabilisation off, focus/exposure locked, all markers + hand visible. Record the S10e as a second view in parallel (unused unless the depth gate below fails). **Also** (~10 min, do regardless of whether the stretch goal below gets attempted): a 2–3 min slow S24 orbit of the table with the ChArUco board on it, plus one short orbit each of the bowl and plate alone — the capture `RealLayout` (§4) needs anyway, and it's the input the splat stretch goal needs if there's time for it later. | ≥ 12 usable clips |
| 09:30–13:00 | **Perception** (`camera`, `markers`, `hand`, `demo`). `02_extract_demos.py` → `HumanDemo` npz + 3D plot per episode. | Clean pick-place arcs, grasp/release detected. **Fallback:** marker on the back of the hand; if that still fails, stereo triangulation from the S10e view (`REAL2SIM_IDEAS.md` §3). |
| 13:00–16:00 | **Twin replay MVP** (`twin`, `sim`, `control`, `retarget`): `SceneTwin` + replay demo #1 in its own real layout, rendered from the real camera viewpoint. **Real ↔ sim-twin side-by-side. Commit it.** | Bowl on plate in ≥ 1 twin replay. **Fallback:** ship the side-by-side anyway with an honest failure note; don't keep debugging past 16:00. |
| 16:00–19:00 | Launch `04_generate_dataset.py` multiprocess on 24 cores (target 300+ successes), log the success rate. Start **ACT** training (~1 h) as soon as there's enough data; SmolVLA only if generation finishes by ~16:30 (needs the longer run in §2's estimate). | Running without errors; loss decreasing after 15 min |
| 19:00–22:30 | **README:** pitch, method, results table, gifs, data-multiplier plot if time allows, what worked/didn't, how to run. Eval (`lerobot-eval`, §6 table). Fresh-clone test (`uv sync` + replay on a clean checkout). Make the repo public. | Clean clone works, numbers in the table |
| 22:30–23:30 | **Buffer, then submit.** Aim to submit by 23:30 CEST (= 22:30 BST), 1.5 h of margin before the hard deadline. | |

**Stretch, gated, optional, no dedicated slot** (`REAL2SIM_IDEAS.md` §7): photoreal real2sim via DISCOVERSE's `gaussian_renderer` (MuJoCo-native, verified on this laptop Fri 02:10 — `gsplat` JIT-compiles against torch cu126, ~2,350 FPS). Train a splat of the bowl+plate+table from the morning's S24 orbit, attach to the LIBERO sim bodies, composite a few twin-replay frames (splat scene, mesh robot, depth-merged). **Only attempted if `04_generate_dataset.py`/training (16:00–19:00) finishes early enough to leave genuine slack before 19:00** — it never borrows time from the README block, and it never starts at all if the 13:00–16:00 MVP gate wasn't met on time. **Hard stop at 19:00 regardless of progress.** Deliverable if it works: a couple of composited frames for the README (LIBERO textures vs. my scene, same pose) — not a retrained policy. If it doesn't get attempted or doesn't produce a clean composite, it becomes a named future-work item in the README instead (cite DISCOVERSE + RoboGSim).

---

## 6. Results to show (the experiment design)

| Policy | Trained on | Success on target task (N = 50) |
|---|---|---|
| `smolvla_base` zero-shot | nothing from LIBERO | expected ~0%, which proves the base model doesn't know the task |
| Open-loop retarget replay | 1 human demo per rollout, no learning | x% |
| **SmolVLA (ours)** | **ego-derived generated demos only** | x% (with the Wilson 95% CI that `lerobot-eval` reports) |
| ACT (ours, fallback/comparison) | same data | x% |
| `smolvla_libero` (reference, if it exists) | official LIBERO teleop demos | upper bound, not our method |

Figures:
- A side-by-side gif: your hand, the retargeted robot, and the learned policy.
- The data-multiplier curve.
- A 3D trajectory plot: human TCP versus robot EEF.
- One honest failure case.
- **If the photoreal stretch goal (§5) was attempted:** one comparison frame, LIBERO's synthetic bowl/plate/table textures vs. the same pose rendered with Gaussian splats of the real objects — presented as a visual-fidelity illustration, not a trained/evaluated result (there is no real robot in this challenge to validate transfer against — see `REAL2SIM_IDEAS.md` §7's framing note).

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| LIBERO install or render fails | 30-minute timebox, then fall back to plain robosuite (§2 gate) |
| MediaPipe loses the hand during the grasp (occlusion) | Interpolate short gaps. A marker on the back of the hand is the robust fallback. |
| Grasps slip in sim | Tune the grasp height and offset per object once; it's fine for that to be a fixed sim-side constant |
| Low generation success rate | Fine as long as there are enough successes in absolute terms. Report the rate honestly as a metric. |
| SmolVLA training too slow or unstable | ACT fallback (~1 h). The story still holds: the policy is trained only on your data. |
| Running out of time | Cut stretch first, then ACT, then the multiplier plot. **Never cut the README block.** |
| Photoreal splat stretch (§5) eats time or the composite looks bad | It has no dedicated slot and a hard 19:00 stop by construction — it literally cannot run past that point. A bad composite just means it becomes a future-work line in the README instead of a figure. |

## 8. Definition of done
A judge clones the repo, runs `uv sync`, and watches a gif of your hand followed by the Panda doing the same task. They read a short method section with a clear "your data → robot demos" multiplier, and they see a results table where a VLA trained **only on data derived from your phone video** succeeds in LIBERO, with an honest note on what didn't work.
