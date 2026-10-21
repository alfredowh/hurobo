# Status and TODO


Legend: ✅ done (the row says whether it ran on real clips or only on stand-in data), 🟡 partly done, ⬜ not started.

| Area | Item | Status |
|---|---|---|
| Setup | uv environment, cu126 torch, LIBERO renders a frame | ✅ |
| Objects | Read the bounding boxes of the LIBERO bowl and plate assets | ✅ (bowl 11.2 cm, plate 13.8 cm; scaled x1.23 and x1.86 to the measured objects, D-010) |
| Objects | Real bowl and plate measured; ArUco pad on the bowl (id 21) and marker on the plate (id 22) | ✅ (measurements in `measurements.yaml`) |
| Capture | Phone calibration (intrinsics files for S24 and S10e exist) | 🟡 |
| Capture | **Real third-person demo clips**, ChArUco board as the table frame, bowl pad and plate marker. Plan, markers and capture check exist ([guide](experiment_scaled_native.md)) | ✅ 26 clips recorded, 12 usable after cleaning |
| Capture | Slow orbit of table, bowl and plate with ChArUco board (for scanning) | ⬜ |
| Perception | ArUco poses, MediaPipe + PnP hand TCP, grasp segmentation, QC report | ✅ |
| Perception | Clip cleaning (`20_clean_clips.py`, notebook 10): per-frame filters, cut to one open-closed-open cycle, clip filters | ✅ on real clips: 12 of 26 usable; 5 of the 12 pass only with the release-distance gate waived |
| Perception | Extraction on real clips. Check the depth anchor correction and the printed marker size | 🟡 runs; the marker size is not measured, so object poses may be scaled wrongly (see the open issues in the [design log](design_decisions.md)) |
| Dataset | `HumanDemo` → LeRobotDataset v3 export and reload | ✅ |
| Sim | `SceneTwin`: real layout and camera in the simulator | ✅ on real clips (notebook 11) |
| Sim | Real and sim-twin side-by-side replay | 🟡 recorded dataset next to robot replay for one clip (notebook 11); that replay timed out; `03_replay.py` is still empty |
| Teleop | Hand-clip and keyboard teleop of the Panda, live window | ✅ |
| Retargeting | `ObjectCentricRetargeter`, fixed top-down orientation, robot from the home pose | ✅ ran on 12 real clips (59 % of rollouts succeed, see generation); first built on made-up hand paths ([D-010](design_decisions.md)) |
| Retargeting | Tune grasp height and offset once | 🟡 success measured over 170 rollouts with default settings; nothing tuned |
| Generation | `DemoGenerator` and `04_generate_dataset.py` (success only, one process) | ✅ first run on 12 cleaned real clips: 100 successes of 170 attempts (59 %); multiprocess not done |
| Generation | MimicGen-style expansion (`22_mimicgen_expand.py`: source screening, nearest-layout sources, action noise, 300-step filter) | 🟡 built; on 7 usable clips it gives 72 to 80 % usable episodes against 15 to 21 % for the default ([optimization potential](extensions.md#optimization-potential-faster-demo-generation), [D-013](design_decisions.md)); not used for the 100-episode training set, and no policy has been trained on it |
| Training | SmolVLA expert-only fine-tune, 3000 steps, batch 16 (`19_train_smolvla.py`, D-013); ACT fallback not done | 🟡 one run, one seed |
| Evaluation | N = 50 with Wilson CI. Baselines: zero-shot, open-loop replay. `lerobot-eval` rebuilds the stock scene, so a loop through `ScaledNativeScene.reset` is needed | 🟡 checkpoints evaluated with N = 10 (`23_eval_checkpoints.py`); comparison with the pretrained checkpoints on 20 held-out layouts done (notebook 14, N = 20); N = 50 and the zero-shot and open-loop-replay baselines not run |
| Figures | GIFs (episode dashboard, dataset and replay, scene orbit, generated clips in the slides), 3D trajectory plot, data-multiplier curve, one failure case | 🟡 GIFs done; the plot, curve and failure case are not |
| Stretch | Splat scan of the real bowl, plate and table. Composited frames. Collision mesh. | 🟡 pipeline built. Real scans of the bowl, the plate and the whole table scene exist (metric ChArUco frame for the scene, [D-009](design_decisions.md); lowered SIFT threshold and floor-based crop, [D-001, D-004](design_decisions.md)), and the scanned bowl and plate are composited into LIBERO ([gallery](extensions.md#what-the-splat-route-has-produced-so-far)). The plate has gaps; the filled version uses a shape prior. One scene, two clips each. The scene splat is not in LIBERO yet, and the collision mesh is not wired into the experiment. |
| Stretch | Second camera view for stereo, only if monocular depth fails | ⬜ |
| Submission | Fresh-clone test, results table, honest "what did not work", public repository | ⬜ |

If time is short, cut in this order: splat stretch, robot overlay, SmolVLA (keep ACT), data-multiplier plot. Never cut the twin side-by-side or this README.
