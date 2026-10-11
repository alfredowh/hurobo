# NovaFlow review: what hurobo can reuse

Reviewed 2026-10-11 against `PLAN.md` (v4) and the current `src/hurobo` modules.

- Repo: <https://github.com/rai-opensource/NovaFlow> (RAI License, ICRA 2026)
- Paper: *NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos*, arXiv:2510.08568 (Li, Sun, Hu, Ta, Barry, Konidaris, Fu)

What I read: the repo README (first-hand) and the paper's HTML version (method, appendix, Tables 1–2). I did **not** read the repo's source files, so statements about what code is actually released are limited to the README's directory listing.

## 1. What NovaFlow does

One correction to the framing first: NovaFlow does not generate a *human hand* video and retarget a hand. A video model generates a video of the **task being solved** (the object moving to its goal). The robot never copies a hand or arm; it copies the **motion of the object**.

```
RGB-D image + instruction (+ optional goal image)
  → Wan2.1 I2V (or Veo 3)                     8 candidate videos, VLM picks the plausible flow
  → MegaSaM + MoGe depth, scale-calibrated    video → metric 3D
  → TAPIP3D                                   dense 3D point tracks (32×32 grid, frame 0)
  → Grounded-SAM2                             keep only tracks on the target object
  → 3D object flow
  → Kabsch/SVD per timestep                   T_obj^t  (rigid objects)
  → T_ee^t = T_obj^t · T_grasp                GraspGen proposes T_grasp from object point cloud
  → PyRoKi / JAX Levenberg–Marquardt          joint trajectory: smoothness, joint limits, swept-volume collision
  → Franka (Robotiq-85) or Spot
```

Deformable objects use the same flow as a tracking cost for a PhysTwin particle model with MPC. This is not relevant to us.

Reported numbers (block insertion, 8 candidate videos per trial, single H100):

| Video source | Video success | Task success | Time |
|---|---|---|---|
| Wan2.1 + goal image | 46% | 80% | ~612 s |
| Wan2.1, no goal image | 15% | 40% | ~612 s |
| Veo 3, no goal image | 75% | 80% | ~20 s |

Authors' own failure analysis: most failures are grasp and execution, not video generation.

## 2. Plan vs. NovaFlow

| Aspect | hurobo (`PLAN.md`) | NovaFlow | Verdict |
|---|---|---|---|
| Source of the motion | **Real** human demos, 15 clips we record | **Generated** video from one image | Opposite on purpose |
| Object pose per frame | ArUco markers (`markers.py`), exact | MegaSaM + TAPIP3D + SAM2, estimated | Ours is cheaper and more accurate |
| Representation | Object-relative segments (approach/carry/place) | Object pose trajectory `T_obj^t`, grasp in object frame | **Same idea** |
| Grasp | Fixed top-down grasp constants per object | GraspGen over a point cloud | See §3 |
| Control | `WaypointFollower` → LIBERO 7-D delta actions | Joint-space trajectory optimization | Not compatible as-is |
| Reject bad rollouts | Keep only sim-verified successes | VLM picks flow before execution | Ours is stronger, since LIBERO gives a ground-truth success check |
| Compute | 16 GB A5500, driver 560.35 | H100 (Wan: multi-GPU recommended) | Does not fit |

Net: hurobo is already a **NovaFlow-style pipeline with real video and markers in place of generated video and learned tracking**. That is a clean framing for the README, and it is honest.

## 3. Component verdicts

| Component | Use it? | Why / how |
|---|---|---|
| **Kabsch/SVD pose from tracked points** | **Yes, adopt the maths** | A few lines in `utils/transforms.py`. Fits `T_table_obj^t` from several tracked points. Use it if the hand-held-object pose comes from more than one marker corner, or to fit a rigid pose to a hand-landmark set instead of the thumb/index midpoint. Needs a test in `tests/`. |
| **`T_ee = T_obj · T_grasp` formulation** | **Yes, adopt as the retargeter's core equation** | `ObjectCentricRetargeter` (planned `retarget.py`) should store the grasp as `T_obj_ee` and the carry as a `T_table_obj` trajectory, then compose against the sim's object pose. Name the frames as in `CLAUDE.md`. |
| **Median-depth scale calibration** | Only if monocular depth enters | Already covered by our anchor-correction (§4.4 of the plan). Same spirit; cite it. |
| **Rejection sampling of candidates** | Already in the plan | `DemoGenerator` keeps only successes. Cite NovaFlow as precedent. |
| **GraspGen** | Optional, low priority | Would replace fixed grasp constants. LIBERO object meshes can supply the point cloud. Costs: a GPU model with its own install on cu126/driver 560.35 (**not verified in this session**), and the plan deliberately fixes top-down grasps. The cheap alternative is a small parametric grasp sampler (yaw ∈ [0, π), height offsets) scored by sim success. Only reconsider if grasps are the measured failure cause. |
| **PyRoKi trajectory optimization** | No | LIBERO is controlled with delta-EEF OSC actions, not joint targets. We would need to FK the joint path back to EEF deltas. Collision avoidance buys little on an open tabletop. |
| **Wan2.1 I2V** | No | Multi-GPU/H100 class; ~12 min per sample on an H100. Will not run on a 16 GB laptop GPU. |
| **Veo 3 via API** | No for the core path | Runs, needs a Google API key, and costs money. More importantly it produces *generated* data, which does not satisfy the challenge's central constraint ("you use data that you personally collected"). A single real photo of the table as the conditioning image is a thin reading of that rule. |
| **MegaSaM, TAPIP3D, Grounded-SAM2** | Stretch only | Their value to us would be a **marker-free** extraction path on our own real clips, as an ablation against ArUco. Heavy installs, GPU-hungry (MegaSaM ~100 s per video on H100). Not worth the risk before the gates in §5 of the plan are met. |
| **Repo code itself** | Reference only | Per README the repo is a Ray client/server wrapper plus Docker and weight scripts; the vendored modules are `tapip3d`, `grounded_sam_2`, `wan2.1`. The README structure does not list the executor (grasp + trajectory optimization), so I could not confirm it is released. Per `CLAUDE.md`, reference repos are never vendored. The RAI License terms were not read; check before copying any code. |
| **PhysTwin / deformables / Spot** | No | Out of scope. |

## 4. Recommendations

1. **Do not integrate the NovaFlow stack.** It targets different hardware and its data source conflicts with the challenge constraint.
2. **Adopt the formulation, not the code.**
   - Add Kabsch/SVD to `utils/transforms.py` with a unit test (only if a multi-point pose fit is actually used).
   - Write the retargeter around `T_table_obj^t` and a fixed `T_obj_ee`.
3. **Cite NovaFlow in the README** alongside Phantom, Video2Sim2Real, Human2Sim2Robot and MimicGen. Suggested honest positioning: *"Where NovaFlow derives object flow from a generated video, hurobo derives it from a real human video via fiducials, then replays and verifies it in simulation."* That states a reuse of the idea and a departure in the data source, which is what the supervisor rules in `CLAUDE.md` ask for.
4. **Optional stretch (only after the MVP replay video and generated dataset exist):** run TAPIP3D + Grounded-SAM2 on one real clip to get marker-free object flow and compare it with the ArUco trajectory. This produces a quantitative figure (trajectory error vs. markers) rather than a claim. Cut it first if the schedule slips.
5. **Optional, only if grasp failures dominate the success analysis:** a sampled-grasp search in sim before considering GraspGen.

## 5. Risks and caveats

- Numbers above are the paper's, from its own hardware and tasks; none were reproduced here.
- The paper text describes the cost terms and hard constraints slightly differently between §3.2.2 and the appendix (start/end configurations are hard constraints in the appendix). Check the PDF if the optimizer formulation is ever reimplemented.
- GraspGen's supported grippers and install requirements were taken from the NovaFlow README's credit only; verify on its own repo before planning around it.
- Deadline is Fri 16 Oct 23:59 BST. Anything beyond item 2–3 above competes with the capture and generation gates in `PLAN.md` §5.

## Sources

- <https://github.com/rai-opensource/NovaFlow>
- <https://arxiv.org/abs/2510.08568>
- <https://arxiv.org/html/2510.08568v1>
