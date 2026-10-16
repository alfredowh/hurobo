# Evaluation results (persistent)

Committed copies of benchmark runs. `results/` is scratch and git-ignored; anything worth keeping is copied here. Produced by `scripts/18_benchmark_policy.py`; read in `examples/09_smolvla_benchmark.ipynb`. Context and caveats: D-011 in `docs/design_decisions.md`.

| Folder | What | Date |
|---|---|---|
| `pretrained_libero_checkpoints/` | `lerobot/smolvla_libero` and `HuggingFaceVLA/smolvla_libero`, closed loop in three scenes (+ one replan-50 cell), open loop on stand-in data | 2026-10-16 |
| `smoke_finetune_20_steps/` | A 20-step fine-tune on 30 stand-in episodes, to prove the train-then-benchmark path. Not a meaningful policy | 2026-10-16 |

Each folder has `closed_loop_episodes.jsonl` (one row per episode), `closed_loop_summary.csv` (success with 95 % Wilson interval), `videos/` (episode 0 of every experiment) and, where run, `open_loop.json` and `logs/`.

## Closed loop, `pretrained_libero_checkpoints`

10 episodes per cell, 300 steps, replan every 10 actions, same layouts for every checkpoint. Scenes: **stock** LIBERO untouched; **clean** other objects removed, LIBERO sizes, random layouts; **scaled** clean + real bowl and plate sizes from `measurements.yaml`.

| Checkpoint | stock | clean | clean, stock layout | scaled | scaled, stock layout | scaled, replan 50 |
|---|---|---|---|---|---|---|
| `lerobot/smolvla_libero` | 10/10 | 0/10 | 6/10 | 0/10 | 0/10 | 0/5 |
| `HuggingFaceVLA/smolvla_libero` | 8/10 | 0/10 | 3/10 | 0/10 | 0/8 (2 episodes missing) | not run |

`*_stock_layout` = the same scene, but bowl and plate start where LIBERO's init state puts them. With LIBERO's placement, clearing the table lowers success (10/10 to 6/10, 8/10 to 3/10) but does not remove it; with random placement it is 0/10. The real-size plate gives 0 even at stock placement (it may overlap the bowl there). Cause not established; n=10.

Zero successes in 10 still allows a true rate up to about 28 %. The cause of the `clean` failure is not established.

## Videos

Episode 0 of each experiment (agent view, 20 fps, caption = scene and episode):

| File (in `videos/`) | Outcome |
|---|---|
| `lerobot_smolvla_libero__stock__10__ep0.mp4` | success, 84 steps |
| `lerobot_smolvla_libero__clean__10__ep0.mp4` | fail |
| `lerobot_smolvla_libero__scaled__10__ep0.mp4` | fail |
| `lerobot_smolvla_libero__scaled__50__ep0.mp4` | fail |
| `HuggingFaceVLA_smolvla_libero__stock__10__ep0.mp4` | success, 83 steps |
| `HuggingFaceVLA_smolvla_libero__clean__10__ep0.mp4` | fail |
| `HuggingFaceVLA_smolvla_libero__scaled__10__ep0.mp4` | fail |
| `../smoke_finetune_20_steps/videos/smoke20__scaled__10__ep0.mp4` | fail |

## Re-run

```bash
MUJOCO_GL=egl uv run python scripts/18_benchmark_policy.py --checkpoints lerobot hfvla \
    --conditions stock clean scaled --replan 10 --episodes 10 --out eval/pretrained_libero_checkpoints
```
