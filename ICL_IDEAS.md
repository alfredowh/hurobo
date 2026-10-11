# In-context learning for ego2robo: review, related work, and other ideas

Written Sat 10 Oct 2026. Status of the sources: everything below was read from arXiv abstracts or the GEN-1.5 blog page in one research pass. I did **not** read full papers, so method details and numbers are only as good as those pages. Anything marked *(from memory)* was not re-checked.

> **Timing note.** `PLAN.md` puts the submission deadline at Fri 9 Oct 23:59 BST, which has passed. If the repo is already submitted, treat this as the follow-up roadmap and the README's future-work section. If there is still a window, §5 gives the smallest version that fits.

---

## 1. What the GEN-1.5 post actually claims

[GEN-1.5 (Generalist AI)](https://generalistai.com/blog/gen-1.5) says a pretrained robot model can learn a new physical task from one or a few demonstrations placed in its context, with no gradient updates ("physical prompting").

- 3–12 s demonstration in context; **59% ± 10%** mean success over 10 simple, short-horizon tasks.
- Two prompts can be chained into one longer behaviour.
- Simulated demos work as prompts for the real robot, although pretraining contained no sim data.
- Gradient adaptation is the stronger mode: 10 steps on ~5 min of data (~50 demos) gives **83% ± 9%**; one step on 1 min gives 66.5% on a held-out task.
- The authors say in-context skills are **more brittle** than fine-tuned ones, and that the ability emerged from scale (8+ months of continuous pretraining), with no architectural change or meta-learning loop.

Two things to take from this for our project:

1. The headline result is an *emergent property of very large pretraining*. We cannot reproduce emergence with a 450M model and a few hundred generated demos. Any ICL we ship has to be **trained in explicitly**, which is what the smaller academic line (§3) does.
2. The blog's own numbers say few-step adaptation beats pure prompting. That supports a second feature (§4, idea A) that costs almost nothing here.

---

## 2. Review of the ICL idea against this project

I could not find the idea written down in the repo, so I am reviewing it as: *"at test time, show the system one human demo of a variant it wasn't trained on, and it performs that variant with no gradient update."*

### What is good about it

- **It matches the data engine we already built.** `generate.py` produces pairs of (human segment, robot rollout from a different init state). A prompt-conditioned policy is trained on exactly that: context = one demo, query = new layout, target = actions. The multiplier story from `PLAN.md` carries over.
- **It makes the human data matter at test time**, not only at training time. That speaks directly to the challenge criterion "the data you collected must play a role" and is a more interesting claim than "SmolVLA reaches x% on one task".
- **Language genuinely cannot specify the task for some variants.** "Put it about 10 cm left of the plate, approaching from the front" is a case where a demo carries information a string does not. Behavior Prompting Policy tests exactly this against language conditioning (§3).

### What is wrong or risky

| # | Problem | Why it matters | Fix |
|---|---|---|---|
| 1 | **One task means nothing to generalise over.** The plan is a single bowl-on-plate task. A policy trained on it learns to ignore the prompt, and "ICL" collapses to plain BC. | Behavior Prompting Policy reports that **task diversity, not demos per task, is the main driver** of prompting ability. | Generate **many goal variants** from the same object-centric segments (§5). |
| 2 | **Copy shortcut.** If the prompt and query come from the same layout, the model copies actions from the prompt. | A high score then says nothing about reading the prompt. | Query layout always differs from prompt layout. Add the control in §5 (shuffled/blank prompt). ICI-VLA also uses a context-corruption objective ("target action masking") for this. |
| 3 | **SmolVLA has no in-context mechanism.** It takes images, state, a language string. Feeding it K demo frames costs context and memory on a 16 GB card, and needs architecture work. | Not doable in a day without risking the working policy. | Don't put the prompt in SmolVLA's context. Use one of the two cheap routes in §5. |
| 4 | **Emergence framing would be dishonest.** | The README is judged for AI slop and unsupported claims. | Say "explicitly trained prompt-conditioned policy, small scale", cite the line of work, and don't borrow GEN-1.5's framing. |
| 5 | **The prompt comes from the retargeter, not the real world, unless you wire it.** A human demo → robot-space prompt goes through the same retargeting that has its own error. | The test-time story ("I showed it once") is only true if a real clip goes through `02_extract_demos.py` → prompt. | Include one real-clip-as-prompt demo and report it as a single qualitative result, not a number. |

### Verdict

**Worth doing, as a scoped feature, not as the headline.** Keep the twin side-by-side and the SmolVLA/ACT results as the core. Add ICL as "can a policy follow a *new* goal from one human demo", with a task family wide enough that the answer isn't trivially yes. Without task diversity, drop it and do idea A or B in §4 instead.

---

## 3. Related work

Verdict column follows the `CLAUDE.md` convention: **reuse** as-is, **modify** for our constraints, or **depart**.

### In-context imitation (robot)

| Work | What it does | Source level | Relevance / verdict |
|---|---|---|---|
| **ICRT**, Fu et al., [arXiv 2408.15980](https://arxiv.org/pdf/2408.15980) (NeurIPS 2024 workshop; [code](https://github.com/Max-Fu/icrt)) | Causal transformer over (image, proprio, action) sequences. A few teleop demos are the prompt; no fine-tuning. Reports 79.2% on unseen pick-place and poking tasks (per alphaXiv summary). | search snippet | Closest simple recipe: **modify**. Same sequence format, trained on our generated data instead of DROID-scale data. |
| **Instant Policy**, Vosylius & Johns, [arXiv 2411.12633](https://arxiv.org/pdf/2411.12633) (ICLR 2025) | ICL as graph generation with diffusion; trained on **simulated pseudo-demonstrations**; learns tasks from 1–2 demos. | search snippet | Strongest precedent that **sim-generated data can train a real ICL policy**: supports our data-engine approach. Graph/point-cloud formulation is heavier than we need; **modify** to a state-based version. |
| **Behavior Prompting Policy (BPP)**, [arXiv 2606.30457](https://arxiv.org/abs/2606.30457) (2026) | One sensorimotor demo as the prompt. New LIBERO-Gen benchmark with up to 2,000 procedurally generated tasks. Finds task diversity drives ICL; weaker than language when diversity is low. Open-sourced; includes a handheld phone-style demo interface (iPhUMI). | abstract | **The most relevant paper.** It is on LIBERO, uses a human demo at test time, and gives us the diversity lesson and a language baseline to copy. Check whether LIBERO-Gen's task generator is reusable before building our own. **Reuse** the evaluation design, **modify** the policy. |
| **ICI-VLA**, Yang et al., [arXiv 2609.07581](https://arxiv.org/pdf/2609.07581) (Sep 2026) | Retrieves in-context micro-demos for a text-action VLM; DTW-mined positives; target action masking. Reports 97.7% on LIBERO. | abstract | Validates ICL on **LIBERO with a VLA**. Retrieval of short, phase-aligned micro-demos is a cheap trick we can borrow: **modify**. Base model unnamed on the abstract page. |
| **ICLR**, Nguyen et al., [arXiv 2603.07530](https://arxiv.org/html/2603.07530v2) (IROS 2026) | Adds image-space "visual reasoning traces" (predicted future trajectories) to demo prompts. | abstract | The image-space trace idea is what makes the SmolVLA-compatible route in §5 plausible: **modify**. |
| **Human demonstration video as a prompt**, Zhu et al., [arXiv 2505.20795](https://arxiv.org/abs/2505.20795) (ICRA 2026) | Human video as prompt; video-generation model learns a joint human/robot representation; real-world dexterous tasks. | abstract (no numbers on page) | Same goal at larger scale and with a learned video model. **Depart**: we use explicit marker/hand geometry instead. |
| **RoboTTT**, [arXiv 2607.15275](https://arxiv.org/abs/2607.15275) (NVIDIA et al., 2026) | Test-time-training layers ("fast weights") scale visuomotor context to 8K timesteps; lists one-shot imitation from human video among capabilities. | abstract | Context *scaling* more than demo prompting; needs GR00T-scale training. **Depart**, cite as the direction large models are taking. |
| **KAT**, Keypoint Action Tokens, [arXiv 2403.19578](https://arxiv.org/html/2403.19578v3) | Writes keypoint observation/action sequences into the prompt of a text-pretrained transformer; ~10 demos, no training. | search snippet | A zero-training baseline worth knowing; object-centric keypoints are the same abstraction as our segments. Optional **modify** as an ablation. |
| **Few-Shot VLA incremental learning**, [arXiv 2504.15517](https://arxiv.org/pdf/2504.15517), **TOPIC** | Learn task-specific prompt vectors from few demos (a gradient-based, not in-context, route). | search snippet | Relevant to idea A below; didn't read in detail. |
| **Generalist survey: "In-Context Learning from Demonstrations for Robotic …"**, [preprint](https://www.preprints.org/frontend/manuscript/786b9fd9a7cbe8084741a434ff3b4c73/download_pub) | 2026 survey; four axes: context information, inference target, adaptation mechanism, evaluated transfer. | search snippet | Good for the README's framing paragraph; not read in full. |

### Human video → robot (already in the plan, listed for completeness)

| Work | Verdict |
|---|---|
| **EgoMimic** ([ResearchGate listing](https://www.researchgate.net/publication/395228678_EgoMimic_Scaling_Imitation_Learning_via_Egocentric_Video)), **EgoVLA**, **EgoVerse** ([arXiv 2604.07607](https://arxiv.org/pdf/2604.07607)), **EasyMimic** ([arXiv 2602.11464](https://arxiv.org/pdf/2602.11464)) | These co-train or pretrain on large human-hand datasets. We are the other end: 15 clips and a simulator. Cite, **depart**. |
| **MimicGen** *(from memory, cited in PLAN.md)* | Object-centric segment retargeting; our generator is a reduced version of it. Already **modified**. |
| **Video2Sim2Real** (arXiv 2606.08828), **Phantom** (arXiv 2503.00779) | Already cited in `PLAN.md` §0 for the twin. |

### Not verified

Search results also mentioned EgoWAM and EgoInfinity only as reference-list entries. I could not confirm their content, so they are not used here.

---

## 4. Ideas beyond in-context learning

Ordered by (value ÷ effort) for *this* repo. Each reuses something already built.

### A. Few-step adaptation instead of (or next to) prompting: **cheapest, do first**
GEN-1.5's own table says gradient adaptation (83%) beats prompting (59%), and a handful of steps suffices once the base model is good. Our analogue: take the SmolVLA/ACT policy trained on the base task, then fine-tune for **N ∈ {0, 10, 50, 200} steps on K ∈ {1, 5, 15} generated demos of a new variant**, and plot success against (steps, demos). It is one plot, it reuses `lerobot-train`, and it honestly answers "how little human data is needed to adapt". It also extends the data-multiplier curve already planned in `PLAN.md` §6.

### B. Hindsight relabeling as the task-diversity engine
A failed place that ends 5 cm left of the plate is a *perfect* demo for "put it left of the plate". Because goals in our pipeline are object-relative, any rollout, failed or not, can be relabeled with the goal it actually achieved. This (i) raises usable-data yield above the generation success rate, and (ii) is exactly the diversity that §2 problem 1 needs. It is HER-style relabeling *(from memory)* applied to the demo generator, with no new learning component.

### C. Visual-trace prompting, so SmolVLA needs no architecture change
Project the prompt demo's TCP path into the policy camera using the camera model `SceneTwin` already has, draw it on the observation image, and fine-tune SmolVLA on the overlaid frames. This is trajectory-conditioned control (RT-Trajectory, ICLR 2024 *(from memory)*) in the spirit of ICLR's image-space traces. Caveat: this conditions on a *path*, which is weaker than a full in-context demo, and must be labeled as such.

### D. Rejection-sampling self-improvement (the challenge's "bootstrap + RL" suggestion)
The generator's success checker is a free verifier. Loop: roll out the trained policy → keep verified successes → add to the dataset → retrain. Report success vs. round. No RL machinery needed; it is expert iteration. Gate it on the base policy's success being nonzero, otherwise there is nothing to filter.

### E. Behaviour-similarity metric against the human
Success rate hides *how* the robot did it. Score each rollout by DTW distance between its object-relative TCP path and the human's. It is nearly free (the 3D trajectory plot is already planned) and gives a second axis: "succeeds *and* moves like the human".

### F. Quantify the real/sim gap with the real marker tracks
The real clips contain actual object trajectories (bowl and plate marker poses over time). Compare them with the sim object trajectories during twin replay. This gives a measured number for what the twin does and doesn't reproduce (slip, release timing), which is exactly the kind of honest limitation a judge rewards.

### G. Challenging embodiments (the brief's first suggestion)
The retargeter is parameterised by object-relative waypoints, so the same human demos can drive a changed gripper (narrower aperture, shifted TCP offset, different speed limits). Show the one-line config change and the success delta. It is cheap, and it directly answers "adapt your data to challenging embodiments".

### H. World-model-lite: object-state forecasting
Skip video prediction. Train a small model to predict the next bowl/plate pose from (current poses, EEF action) on generated sim data, then check one-step error on the real marker tracks (F). It touches the brief's "world model" option at a scale that fits, and it is a different claim from the policy, so it doesn't compete with it. Honest scoping: this is *state* prediction, not video.

---

## 5. Recommended minimal ICL scope (if there is time)

**Goal:** one claim, tested properly: *a policy conditioned on one retargeted demo does the demonstrated variant in a held-out layout, and does worse with the prompt removed.*

1. **Task family (the part that makes it ICL).** Parameterise the place goal as an object-relative offset from the plate: {on, left-of, right-of, in-front-of, behind} × {2 distances}. All variants share the same grasp/carry segments, so the generator needs a new *place target*, not new perception. Use B to relabel failures into the nearest variant.
2. **Prompt format.** One demo = its object-relative waypoints (TCP pose in the bowl/plate frame, gripper state), downsampled to ~16 tokens. State-based, not pixels.
3. **Policy.** Small transformer: prompt tokens + current object poses + EEF state → next action chunk. Train from scratch on generated data; query layout always differs from the prompt's layout. Add target-action masking *(ICI-VLA)* only if the shuffled-prompt control shows copying.
4. **Be explicit that this is not SmolVLA.** If a pixel-based version is wanted, use route C (trace overlay) and keep SmolVLA unchanged.
5. **Controls (all required, otherwise the claim is not supported):**
   - correct prompt vs. **blank prompt** vs. **shuffled prompt** (prompt from a different variant);
   - **held-out variant** (e.g. train on 8 of 10 offsets, test on 2);
   - language-only baseline for the same variants, as in BPP;
   - N ≥ 50 episodes per cell with a Wilson interval, as in `PLAN.md` §6.
6. **One qualitative real-clip demo:** a new real phone clip → `02_extract_demos.py` → prompt → policy in the twin. Reported as a single example, not a number.

**Gate:** the correct-prompt success must beat the shuffled-prompt success by a clear margin on held-out variants. If not, the policy ignores the prompt: report that as a negative result and fall back to idea A. **Time box:** half a day for generator variants + training; no dedicated slot should displace the README.

**Honest claim to write if it works:** "At small scale, with explicitly trained prompt conditioning on sim-generated data, a policy followed a one-demo goal specification in held-out layouts. This is not the emergent ICL GEN-1.5 describes, which arises from large-scale pretraining."

---

## 6. If only one thing gets done

Do **A** (few-step adaptation curve). It takes the least engineering, reuses `lerobot-train` and the generator unchanged, answers a question GEN-1.5 itself raises, and the result is a plot rather than a claim that needs five controls.

## Sources

- [GEN-1.5 — Generalist AI](https://generalistai.com/blog/gen-1.5)
- [ICRT — In-Context Imitation Learning via Next-Token Prediction](https://arxiv.org/pdf/2408.15980)
- [Instant Policy — In-Context Imitation Learning via Graph Diffusion](https://arxiv.org/pdf/2411.12633)
- [Behavior Prompting Policy](https://arxiv.org/abs/2606.30457) · [project page](https://behavior-prompting.github.io/)
- [ICI-VLA](https://arxiv.org/pdf/2609.07581)
- [ICLR — In-Context Imitation Learning with Visual Reasoning](https://arxiv.org/html/2603.07530v2)
- [Human Demonstration Video as a Prompt](https://arxiv.org/abs/2505.20795)
- [RoboTTT](https://arxiv.org/abs/2607.15275)
- [Keypoint Action Tokens](https://arxiv.org/html/2403.19578v3)
- [Few-Shot VLA incremental policy learning](https://arxiv.org/pdf/2504.15517)
- [EgoVerse](https://arxiv.org/pdf/2604.07607) · [EasyMimic](https://arxiv.org/pdf/2602.11464)
