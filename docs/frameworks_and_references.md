# Frameworks, models and references

## Frameworks and models


| Component | Choice | Reason |
|---|---|---|
| Simulator and benchmark | **LIBERO** (`hf-libero`, on robosuite and MuJoCo) | It is the simulator in the challenge brief. It installs with pip. It has a Panda arm. `lerobot-eval` supports it. Generation and evaluation use one env wrapper, so observations and actions match. |
| Policy framework | **LeRobot** (0.6.1 or later) | It gives `lerobot-train`, `lerobot-eval`, the LIBERO wrapper and the LeRobotDataset v3 format. |
| VLA | **SmolVLA** (`lerobot/smolvla_base`, 450M) | It fits the 16 GB GPU in bf16. The checkpoint has not seen LIBERO, so prior task knowledge does not confuse the result. |
| Fallback policy | **ACT** | It trains from scratch in about one hour. It is strong on single-task LIBERO. |
| Hand tracking | **MediaPipe Hand Landmarker** | It runs on CPU. It gives 2D landmarks and metric world landmarks. `solvePnP` converts them to a metric hand pose without a depth sensor. |
| Frames and objects | **OpenCV ArUco** | It gives 6-DoF poses without machine learning. It is robust. The markers do not appear in the simulator. |
| Intrinsics | **ChArUco** calibration (OpenCV) | Correct PnP and the simulator camera FOV need the intrinsics. |
| Scene reconstruction | **pycolmap** (SfM) and **gsplat** (Gaussian splatting) | They estimate poses and train a splat from a phone orbit video. They run on this GPU with the cu126 build. |
| Splat rendering | **DISCOVERSE `gaussian-renderer`** | It is a MuJoCo-native splat renderer. It composites with mesh geometry by depth. |
| Collision geometry | **open3d** and **CoACD** | They convert a splat to a surface mesh and convex pieces. Scanned objects then have a solid body. |
| Tooling | **uv**, **ruff**, **pytest**, Jupyter and **plotly** | Reproducible environment, formatting and lint, tests, interactive 3D views. |

Related work (full citations in the References section below): MimicGen [4] (used as a concept), HaWoR [5] (considered and not used, because the camera is fixed), NovaFlow [6], Video2Sim2Real [7], Phantom [8], PolaRiS [9] and RoboGSim [10] (human-video and real2sim ideas, not used as code).

NovaFlow [6] is the closest in spirit and differs in the source of the motion. It turns a video made by a generative model into 3D object flow and executes that flow on a robot without any training. This project uses a real video of a human hand, retargets the hand path to many simulated object poses, and trains SmolVLA on the verified runs. I do not reuse NovaFlow's code or models, and I have not compared against it.

## References


Papers behind the components and related work in the first section. The status of each (as-is, configured, modified, rejected) is in the [design log](design_decisions.md). Software without a paper is listed under "Software".

### Related work

1. M. Shukor et al. **SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.** 2025. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
2. B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, P. Stone. **LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.** NeurIPS Datasets and Benchmarks, 2023. [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
3. T. Z. Zhao, V. Kumar, S. Levine, C. Finn. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** (ACT). RSS, 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
4. A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, D. Fox. **MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations.** CoRL, 2023. [arXiv:2310.17596](https://arxiv.org/abs/2310.17596)
5. J. Zhang, J. Deng, C. Ma, R. A. Potamias. **HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos.** CVPR, 2025. [arXiv:2501.02973](https://arxiv.org/abs/2501.02973)
6. H. Li, L. Sun, Y. Hu, D. Ta, J. Barry, G. Konidaris, J. Fu. **NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos.** 2025. [arXiv:2510.08568](https://arxiv.org/abs/2510.08568)
7. J. Qiu, Y. Han et al. **Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single Human Video.** 2026. [arXiv:2606.08828](https://arxiv.org/abs/2606.08828)
8. M. Lepert, J. Fang, J. Bohg. **Phantom: Training Robots Without Robots Using Only Human Videos.** CoRL, 2025. [arXiv:2503.00779](https://arxiv.org/abs/2503.00779)
9. A. Jain, M. Zhang et al. **PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies.** 2025. [arXiv:2512.16881](https://arxiv.org/abs/2512.16881)
10. X. Li, J. Li et al. **RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator.** 2024. [arXiv:2411.11839](https://arxiv.org/abs/2411.11839)

### Components

11. J. Wu et al. **DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments.** IROS, 2025. [arXiv:2507.21981](https://arxiv.org/abs/2507.21981)
12. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. **3D Gaussian Splatting for Real-Time Radiance Field Rendering.** ACM TOG (SIGGRAPH), 2023. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
13. V. Ye et al. **gsplat: An Open-Source Library for Gaussian Splatting.** 2024. [arXiv:2409.06765](https://arxiv.org/abs/2409.06765)
14. J. L. Schönberger, J.-M. Frahm. **Structure-from-Motion Revisited** (COLMAP). CVPR, 2016. [PDF](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)
15. F. Zhang et al. **MediaPipe Hands: On-device Real-time Hand Tracking.** 2020. [arXiv:2006.10214](https://arxiv.org/abs/2006.10214)
16. X. Wei et al. **Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search** (CoACD). SIGGRAPH, 2022. [arXiv:2205.02961](https://arxiv.org/abs/2205.02961)

### Software

- [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face): training, evaluation and the LeRobotDataset v3 format.
- [robosuite](https://robosuite.ai) and [MuJoCo](https://mujoco.org): the physics and rendering under LIBERO.
- [OpenCV](https://opencv.org) (ArUco, ChArUco, `solvePnP`) and [Open3D](https://www.open3d.org).

Author lists marked "et al." are shortened; the arXiv page of each entry has the full list. Every title and arXiv number was checked against its arXiv listing. I read the abstracts only, not the full papers, for entries 6 to 11.
