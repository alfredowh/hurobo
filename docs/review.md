# Review: what the results do and do not show

This review applies the standard of a robot-learning engineer who needs a result that holds in production. The scope is the claim in the README: human video improves a small VLA on a new layout and a new object.

**Verdict.** The engineering is good. The scientific claim is not yet shown. The main reason is that the experiments have no control without human data. Production relevance is limited.

## 1. The central claim is not isolated

- No condition without human data was run.
- The human video gives only the shape and timing of the approach and carry paths. The robot sets grasp height, finger yaw and release height.
- A hand-coded planner can make a path of this kind. The challenge text says the task can be hand-coded.
- The control experiment: use the same generator with a synthetic path (straight line or min-jerk), the same 100 episodes, the same fine-tune and the same held-out layouts. Compare both policies.
- Until this runs, the data is shown to be an input. It is not shown to be necessary.

**Counter-argument.** The control can give an equal result. Four reasons why the human data is still of value in that case:

1. **The test covers the weakest part.** The control tests only the path shape. The human also gives the object poses and sizes, measured with markers. The human also shows the task steps: grasp at the rim, carry, release over the plate. A planner that works needs a designer who knows these steps. The human data replaces this design work. It does not have to give a better result.
2. **The task is too easy to show a difference.** Bowl on plate with a fixed top-down grasp is solved by a straight-line planner. A null result is expected. The human path can only matter on tasks that are hard to script, for example cluttered scenes or odd objects. A null result on an easy task does not show that the method has no value. It shows that the task cannot separate the two sources. This is a claim only. A harder task must prove it.
3. **Human paths contain natural variation.** Speed, curvature and timing differ between clips. A synthetic path is one smooth curve. This variation can help robustness, also when the mean success is equal. Success rate at N = 20 is a coarse measure. Better measures: time to complete, failure modes, and robustness to a pushed bowl or a noisy layout. This is a hypothesis. It is not measured.
4. **The human is the more scalable source.** A scripted planner needs a new script for each task. A human only does the task. For one task the cost is equal. For many tasks it is not. In this case the argument is cost, not quality. Today the pipeline needs markers on each object and a ChArUco board, so it is not yet cheaper than a script.

If the control gives an equal result, the honest position is to report it, and to name a harder task as the next test.

## 2. The headline comparison is weak

- The pretrained checkpoints (0/20) never saw the scaled scene. A fine-tuned model that wins against them shows only that training on the target scene helps.
- Three baselines are missing: zero-shot `smolvla_base`, open-loop replay of the retargeted paths, and a policy trained on the generator without human data.

## 3. No test of the VLA part, no ACT comparison

- There is one task and one instruction. Language is never changed, so the language and vision-language parts of SmolVLA are not tested.
- ACT is a much smaller policy that trains from scratch. It is strong on single-task LIBERO. It could reach the same success on this data.
- The ACT baseline is not run. Without it, the result cannot show that a VLA is needed for this task.
- The test to run: ACT on the same 100 episodes, the same held-out layouts and the same budget. Report success, training time and inference speed.
- If ACT matches SmolVLA, the VLA adds cost and no benefit here. If SmolVLA wins, that is a real argument for the VLA.

## 4. The base-init and libero-init comparison is confounded

- Training is 3000 steps at batch 16. This is about 1.35 epochs over 35k frames. The vision encoder is frozen.
- The `smolvla_base` run (3/20) is possibly under-trained. The result may not show that LIBERO priors help.
- The checkpoint curve of the libero-init run (2, 1, 7, 5, 8, 7 of 10) shows high noise at N = 10.
- The best of six checkpoints was selected. Only one seed was run.
- Test: train `smolvla_base` for 10 000 to 20 000 steps and compare again.

## 5. "New object" is an overclaim

- The objects are the LIBERO bowl and plate meshes, scaled by 1.23 and 1.85. They are not new objects.
- Test layouts come from the same generator as the training layouts. Camera, instruction and objects are the same.
- The result shows interpolation inside one distribution. It does not show generalisation to new objects or new scenes.

## 6. Data quality of the core signal

- The printed marker size was not measured. Object poses in the table frame can be scaled wrongly. All object-relative numbers depend on this.
- 5 of the 12 clips pass only with the release-distance gate waived.
- The generator succeeds in 59 % of rollouts. The median episode length is 321 steps. The evaluation limit is 300 steps. No length filter was applied.
- The share of the 100 episodes from each of the 12 clips was not checked.
- All clips come from one person, one session and one object pair.
- The pipeline needs a ChArUco board and a marker on each object. It cannot use video from an uncontrolled setting. This limits the use in production.
- The grasp is fixed top-down. All hand orientation is discarded. All results are in simulation only.

## What is credible

- The held-out protocol: the checkpoint is selected on layouts 0 to 9 and tested on layouts 100 to 119.
- Wilson intervals, and failed runs that are reported.
- The negative results in [design_decisions.md](design_decisions.md).
- The measured layout sensitivity of the pretrained checkpoints (0/10 on a new layout). See [results.md](results.md).

## Next experiments, by value for effort

1. No-human ablation (see point 1).
2. Open-loop replay baseline. It needs no training. It gives the ceiling of the generator.
3. ACT on the same data (see point 3).
4. `smolvla_base` with a longer run (see point 4).
5. Measure the marker size with a ruler. Extract the clips again.
6. One fixed evaluation for the final rows: 300 steps, N = 50, three seeds.
