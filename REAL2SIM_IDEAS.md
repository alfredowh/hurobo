# Real2Sim with a third-person phone camera: idea review

Written Fri 9 Oct, 00:50 CEST. The deadline is **23:59 BST = 00:59 CEST Saturday**, so about 24 h remain. The repo has docs only and no code yet.
This file reviews four ideas: a third-person view, real2sim with the hand replaced by a robot, a phone on the wrist as an IMU, and the smartwatch. It also says how far each one can be combined with `PLAN.md`. Nothing in `PLAN.md` changes unless we adopt something from here.

---

## TL;DR (supervisor verdict)

| Idea | Verdict | Why |
|---|---|---|
| **Third-person view (S24 on a tripod)** | **Adopt. `PLAN.md` already uses it** | The "ego" in the plan is a naming leftover. The plan already puts the phone on a tripod. A static camera is what makes real2sim cheap: the ArUco table marker gives the camera pose once, and the sim camera can be put in exactly the same place. |
| **Real2sim: the hand is replaced by a robot in a sim copy of my table** | **Adopt as the headline.** Upgrade `PLAN.md` §4 from "motion retargeting" to a **digital-twin replay** | We already measure everything needed: object poses (ArUco), camera pose and intrinsics (ArUco + ChArUco), and the hand TCP (MediaPipe + PnP). Copying them into the sim costs a few hours and gives a much stronger video: *my real table → the same table in sim, same viewpoint, with a Panda where my hand was.* |
| **Drop LIBERO, build my own sim** | **Keep LIBERO/robosuite as the engine.** Only reconfigure the scene | LIBERO already has a bowl, a plate, a Panda, a working controller and `lerobot-eval`. The real2sim part is *placing* those objects and a camera from real measurements. That needs no new engine. A custom MuJoCo scene (menagerie Panda) costs about half a day of env + eval plumbing and gives nothing a judge can see. |
| **S10e strapped to the hand as an IMU** | **Don't use raw IMU for position.** Better roles for the S10e, in order: (1) **second camera for stereo**, (2) ArCore 6-DoF wrist pose (only post-deadline) | Integrating accelerometer data twice drifts by tens of cm within seconds. Orientation from the gyro is fine, but the plan fixes the gripper orientation top-down anyway. A 150 g phone on the back of the hand also hides the hand from MediaPipe. |
| **CMF Watch Pro 2 GPS** | **Drop** | GPS is accurate to metres and gets no fix indoors, while the task spans 30 cm. The watch runs Nothing's own OS (not Wear OS): no third-party watch apps and no raw sensor access. Even workout GPS tracks can't be exported cleanly (see below). |

**Recommended combined storyline:**
> *"I recorded myself moving a bowl onto a plate with one phone on a tripod. The pipeline rebuilds my table in simulation (same object layout, same camera viewpoint) and replaces my hand with a Panda that replays my motion. It then multiplies those demos MimicGen-style and trains a policy only on them."*

This keeps every gate and fallback in `PLAN.md`. It adds one cheap, highly visible deliverable, the **real | sim-twin side-by-side from the same viewpoint**, plus an optional overlay of the robot on the real video.

---

## 1. Third-person vs. egocentric

| | Egocentric (head or chest phone) | Third-person (tripod) |
|---|---|---|
| World frame | Camera moves, so you need SLAM to get hand motion in a world frame (that is HaWoR's whole job) | Camera is static, so one ArUco detection fixes `T_cam_table` for the whole clip |
| Sim camera | Can't be reproduced in sim (it moves with the head) | `T_table_cam` + intrinsics → place a MuJoCo camera at the same pose and FOV and render the matching view |
| Hand visibility | Fingers are seen from behind and above; good for grasp aperture | The back of the hand can hide the fingertips at the grasp. Mitigate by putting the camera about 45° to the side, raised, looking down |
| Matches the brief? | The brief's example says "egocentric", but it explicitly invites any creative use of your own phone data | Fine. Say in the README that the third-person view is a deliberate choice that enables the digital twin |

**Decision:** third-person tripod (already in the plan). Rename the pitch from "ego video" to "phone video" so the README isn't inconsistent.

---

## 2. Real2sim: what it means here, and prior work

"Real2sim" covers two different things. We should do the first fully and the second only partly.

**(a) Physical/scene real2sim (digital twin).** Copy the real scene's geometry into the simulator, then let the robot do what the hand did.
- **Video2Sim2Real** (arXiv 2606.08828, Jun 2026): from a single human video, it uses foundation models to build a sim-ready digital twin, extracts object-centric keyframes, and anchors robot configurations to them. *This is the closest prior work to our idea.* Its object-centric keyframe anchoring is the same move as our anchor correction (`PLAN.md` §4.4).
- **Human2Sim2Robot** ("Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration", arXiv 2504.12609): one RGB-D human video → object pose trajectory + pre-grasp pose → sim RL.
- **Track, Articulate, Act** (arXiv 2609.19119): retargets a reconstructed hand into MuJoCo and replays it open loop.
- **Reuse vs. modify:** these papers rebuild object *meshes* with foundation models (and often need depth). We **modify**: we skip mesh reconstruction, because LIBERO's bowl and plate are good enough stand-ins (`PLAN.md` §1a). We copy only **poses + camera** from the real scene. Fiducials replace the foundation models, which is cheaper and deterministic, and the S24 has no depth stream anyway.

**(b) Visual real2sim / "robotizing" the video.** Paint the robot over the human in the real video.
- **Phantom** (Lepert, Fang, Bohg, CoRL 2025, arXiv 2503.00779): estimates hand pose, inpaints the hand away, renders a robot in its place, and trains on the edited video. Reports up to 92% real-robot success. It uses an RGB-D camera.
- **Masquerade** (same group, arXiv 2508.09976): the same editing applied at scale to in-the-wild egocentric video.
- **Reuse vs. depart:** we **don't** train on edited real frames. Our policy trains on sim frames (`PLAN.md` §1a, storyline A). We use the overlay only as a **visualisation / sanity check**: the rendered Panda TCP should sit on my pinch point in every frame. That is a visual check of the whole calibration chain (intrinsics, `T_cam_table`, hand PnP) that a judge understands at a glance. The full Phantom-style inpainting is out of scope.

**Our novel twist (state it plainly in the README):** we use the same tripod camera for three things: (1) metric perception of the hand and objects, (2) instantiating the sim twin (object layout *and* sim camera), and (3) checking the reconstruction by reprojecting the robot onto the real video. None of the cited works runs this loop with one consumer phone and printed markers.

### What changes in the pipeline (add-ons to `PLAN.md` §3–4)

| New piece | Where | What it does | Cost |
|---|---|---|---|
| `SceneTwin` | `src/ego2robo/twin.py` | From the first frames of a clip: `T_table_bowl`, `T_table_plate`, `T_table_cam`, intrinsics → sets the bowl/plate free-joint qpos in the LIBERO sim (table-marker origin mapped to a fixed point on the sim table) and moves a **non-policy** camera to `T_world_cam` with `cam_fovy` taken from `fy`. `mjModel.cam_pos`, `cam_quat` and `cam_fovy` can be written at runtime, so no XML editing is needed. Check the available names with `sim.model.camera_names` and leave `agentview` (the policy camera) untouched. | ~2 h |
| Twin replay | `scripts/03_replay.py --twin` | Replays demo *k* in its **own** real layout, not a LIBERO init state, and renders from the phone's viewpoint → `real | twin` side-by-side at the same angle | ~1 h on top of the MVP replay |
| Overlay (stretch) | `utils/video.py` | MuJoCo segmentation render of the robot → alpha-composite onto the real frame at 50% opacity. Don't inpaint the hand: keeping it visible *is* the check. | ~1 h |

Frame-convention traps to write down in code comments: OpenCV camera looks down +z with y pointing down, while MuJoCo cameras look down −z with y pointing up. So `R_mj = R_cv · diag(1, −1, −1)`. Also check whether the MuJoCo render is flipped vertically (LIBERO flips images in the LeRobot wrapper).

Sim–real scale: the LIBERO table area is about the size of a real kitchen table area. Lay out the real bowl and plate inside the region the Panda can reach (~0.3–0.7 m in front of the base) so that copying the poses 1:1 stays reachable. **If a real layout is out of reach, clamp it and log that. Don't silently rescale it.**

---

## 3. S10e on the hand: IMU / ARCore research

**Raw IMU (accelerometer + gyroscope), e.g. phyphox or OpenCamera Sensors → CSV:**
- Position from double-integrating the accelerometer drifts by tens of cm within a few seconds (well-known; this is why every system below uses vision-inertial fusion, not raw IMU).
- Orientation (gyro + gravity) is good for roll and pitch, and yaw drifts slowly. **But our retargeting fixes the gripper top-down with yaw taken from the object**, so wrist orientation isn't used.
- **What raw IMU could give:** (i) a time-sync spike (a clap or tap), (ii) a sharper grasp/release time from the jerk at contact. Both are already solvable from video (aperture with hysteresis).
- Related work: **IMUPoser** (CHI 2023, arXiv 2304.12518) and **MobilePoser** (arXiv 2504.12492) estimate body pose from IMUs in phones, watches and earbuds. **SmartPoser** adds UWB ranging between phone and watch for arm pose. They work at body scale with errors of several cm or more, which is too coarse to place a gripper on a bowl.

**ARCore visual-inertial odometry (the version that actually works):**
- **Phone2Act** (arXiv 2605.01948): ARCore 6-DoF pose at 50 Hz for VLA teleop data. **COIN** (arXiv 2604.16886): ARKit/ARCore phone teleop at 20 Hz. **TeleDex** (arXiv 2603.17065): phone mounted rigidly on the hand, with ARKit wrist pose plus hand landmarks from the front camera. **HoMMI** (arXiv 2603.03243): iPhones on grippers, ARKit poses at 60 Hz.
- So "phone on the wrist" is an established, working recipe, **but only with VIO**, not raw IMU.
- Tooling on Android: `PyojinKim/arcore-data-logger` logs ARCore 6-DoF poses to a text file, but it needs a build in Android Studio 3.4 / API 28 (2019-era toolchain) and records no video. **MARS logger** (OSU PCVLab) records frames + IMU with good sync. Building and debugging either one tonight would eat the capture block.
- Practical problems with our setup: the S10e on the back of the hand **hides the hand from the S24's MediaPipe** and changes how you grasp. Its pose then has to be put into the table frame with a calibration offset, plus clock sync between the two phones.

**Verdict:** not on the critical path. The cheaper way to get the same benefit (a robust 6-DoF wrist pose) is already the plan's fallback: **a printed ArUco marker on the back of the hand**, seen by the S24. Same frame, same clock, zero sync.

### Better use of the S10e: a second tripod camera (optional upgrade, gated)
- The weakness of the plan is **monocular depth** (PnP on MediaPipe world landmarks, then anchor correction).
- With the S10e as a second static view about 60–90° from the S24: both phones see the table marker → `T_cam1_table` and `T_cam2_table` → **triangulate** the 2D thumb and index tips with `cv2.triangulatePoints`. That gives metric 3D with no PnP depth ambiguity, and each camera also covers the other's occlusions at the grasp.
- Sync: one sharp clap at the start of each take (audio peak in both files), refined by matching the first frame where the hand enters. At 30 fps the residual offset is ≤ 17 ms, which is negligible at hand speed.
- Cost: ChArUco calibration of the second phone (10 min), sync code (~1 h), triangulation (~1 h). **Gate:** only if the monocular trajectories fail the perception gate in `PLAN.md` (noisy depth), and only if the marker-on-hand fallback also fails. Record with both phones anyway: it costs nothing at capture time and keeps the option open.

---

## 4. CMF Watch Pro 2 (Nothing)

- **OS:** Nothing's own RTOS, **not Wear OS**. No third-party watch apps, no developer SDK, no raw accelerometer/gyroscope stream.
- **GPS:** built-in multi-system GNSS, used for workouts only. Data leaves only through the CMF Watch app → **Strava / Google Health Connect / Apple Health**. Nothing community threads report that Strava uploads often arrive **without the GPS track** (as a plain "Workout"), and there is **no GPX export** (it's a user feature request).
- **Even with perfect access:** consumer GNSS is accurate to about 3–5 m under open sky and much worse or unavailable indoors. The task moves the hand about 0.3 m. GPS is 10–100× too coarse and doesn't work at a desk.
- **What the watch could contribute:** nothing useful for manipulation. At most the heart rate (irrelevant here). **Drop it**, and mention it in one line of the README under "considered and rejected", because that shows judgment.

---

## 5. Hardware & capture setup (all owned, plus a printer)

| Item | Role | Settings / notes |
|---|---|---|
| **Galaxy S24** | Main third-person camera | Tripod or a stack of books, ~45° to the side and raised, looking down at the table. **Pro Video** mode: 1080p30, manual focus + exposure locked, **video stabilisation / Super Steady OFF** (stabilisation warps the frames and breaks the intrinsics). Calibrate with ChArUco **at the same zoom/lens and resolution** you record with. |
| Galaxy S10e | Optional 2nd view (stereo), same settings | Calibrate separately. Clap sync. |
| CMF Watch Pro 2 | — | Not used. |
| Printed ArUco markers | Table frame (≥ 10 cm), bowl and plate (~4 cm), optional back-of-hand (~5 cm) | DICT_4X4_50, measure the printed side length with a ruler, mount flat. |
| Real bowl + plate | The task | Roughly the size of the LIBERO assets. Plain colours help MediaPipe. |
| Laptop (RTX A5500 16 GB, 24 cores) | Everything else | MediaPipe on CPU, sim on CPU, SmolVLA/ACT on GPU. **All of this fits on this laptop**: no cloud and no mesh reconstruction needed. |
| USB-C cable | Transfer | `adb pull /sdcard/DCIM/Camera/...` or Quick Share. Keep the originals and never re-encode them before calibration. |

Not needed: a depth camera, FoundationPose (needs RGB-D or a mesh plus depth for the S24 view), SAM 3D Objects / photogrammetry (LIBERO assets stand in for the objects, see `PLAN.md` §1a), HaMeR or WiLoR (better hands, but GPU models with heavier setup. **Reserve WiLoR as the replacement if MediaPipe loses the hand at the grasp.**)

---

## 6. Revised critical path (now that it's 00:50 Friday with no code)

The Thursday blocks in `PLAN.md` have all slipped. Proposed re-time (times in CEST; the hard deadline is 00:59 Sat):

| Time | Block | Gate → fallback |
|---|---|---|
| now → 02:00 | `uv` setup + `00_check_env.py` (LIBERO renders). Print the markers. **Then sleep.** | LIBERO frame saved → otherwise plain robosuite tomorrow |
| 08:00–09:30 | Capture: ChArUco calibration + 15 takes on the S24 (S10e second view recording in parallel, unused unless needed) | ≥ 12 usable clips |
| 09:30–13:00 | Perception (`camera`, `markers`, `hand`, `demo`) | Clean pick-place arcs → otherwise the marker-on-hand track |
| 13:00–16:00 | **Twin replay MVP**: `SceneTwin` + replay in the real layout + same-viewpoint side-by-side. **Commit.** | Bowl on plate in ≥ 1 twin replay → otherwise ship the side-by-side anyway with an honest failure note |
| 16:00–19:00 | Generation (multiprocess) → **ACT** training (~1 h). SmolVLA only if generation finishes by 16:30. | Numbers in the table |
| 19:00–22:30 | README, gifs, eval table, fresh-clone test, push public | — |
| 22:30–00:59 | Buffer. **Submit by 23:30 CEST.** | — |

Order of cuts if late: robot overlay → SmolVLA (keep ACT) → data-multiplier plot → stereo. **Never cut** the twin side-by-side or the README.

---

## Sources
- Phantom: [arXiv 2503.00779](https://arxiv.org/abs/2503.00779), [CoRL 2025 / PMLR](https://proceedings.mlr.press/v305/lepert25a.html)
- Masquerade: [arXiv 2508.09976](https://arxiv.org/abs/2508.09976)
- Video2Sim2Real: [arXiv 2606.08828](https://arxiv.org/pdf/2606.08828)
- Human2Sim2Robot: [arXiv 2504.12609](https://arxiv.org/abs/2504.12609)
- Track, Articulate, Act: [arXiv 2609.19119](https://arxiv.org/pdf/2609.19119)
- IMUPoser: [arXiv 2304.12518](https://arxiv.org/pdf/2304.12518) · MobilePoser: [arXiv 2504.12492](https://arxiv.org/html/2504.12492v1)
- TeleDex: [arXiv 2603.17065](https://arxiv.org/pdf/2603.17065) · Phone2Act: [arXiv 2605.01948](https://arxiv.org/pdf/2605.01948) · COIN: [arXiv 2604.16886](https://arxiv.org/pdf/2604.16886) · HoMMI: [arXiv 2603.03243](https://arxiv.org/pdf/2603.03243)
- ARCore Data Logger: [github.com/pyojinkim/arcore-data-logger](https://github.com/pyojinkim/arcore-data-logger) · MARS logger: [OSUPCVLab wiki](https://github.com/OSUPCVLab/mobile-ar-sensor-logger/wiki) · OpenCamera Sensors: [prime-slam/OpenCamera-Sensors](https://github.com/prime-slam/OpenCamera-Sensors)
- CMF Watch Pro 2 export limits: [Nothing community: Strava integration](https://nothing.community/d/12851-cmf-watch-pro-2-strava-integration), [improved Strava integration request](https://nothing.community/d/39668-improved-strava-integration-for-cmf-watch), [CMF Watch app third-party sync](https://www.thetechoutlook.com/news/os/cmf-watch-app-gets-a-new-update-with-improved-exercise-data-sync-and-support-for-third-party-apps/), [Galaxus review (not Wear OS)](https://www.galaxus.de/en/page/cmf-watch-pro-2-tested-excellent-smartwatch-but-without-nfc-33840), [GSMArena launch (GNSS)](https://www.gsmarena.com/cmf_watch_pro_2_and_buds_pro_2_announced-news-63611.php)
- SAM 3D Objects: [arXiv 2511.16624](https://arxiv.org/pdf/2511.16624) · obj2mjcf/CoACD: [github.com/kevinzakka/obj2mjcf](https://github.com/kevinzakka/obj2mjcf)
- HaMeR: [arXiv 2312.05251](https://arxiv.org/html/2312.05251v1) · WiLoR: [arXiv 2409.12259](https://hyper.ai/en/papers/2409.12259) · FoundationPose: [NVlabs](https://research.nvidia.com/publication/2024-06_foundationpose-unified-6d-pose-estimation-and-tracking-novel-objects.md)
