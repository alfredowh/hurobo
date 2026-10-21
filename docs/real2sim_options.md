# Real2sim: how to close the gap


The central question: how much of the real world must the simulator copy so that a policy trained from human video works in the simulator? This section compares four solutions. They go from "copy nothing" to "copy the whole scene".

### Solution 1: teleoperate directly in the simulator

Collect the data in the simulator. Control the Panda with a tracked hand or with the keyboard.

- **Distribution shift:** none for visuals. Training and test use the same renderer.
- **Problem:** it is hard to control a 7-DoF arm precisely with a tracked hand or a keyboard. The demos are slow, jerky and few. The real recording has no role. This breaks the challenge rule that the recorded real data must be used.
- **Status:** built as a tool (`teleop.py`, `13_teleop.py`, notebook 05). It is for validation and debugging. It is not the data source.

### Solution 2: scan the real objects and add them to the simulator

Reconstruct the bowl and plate from a phone orbit video. Place them in the LIBERO scene.

- **Idea:** the objects are what the policy touches. If they look and collide like the real objects, the most important part of the visual gap closes. The rest of the scene stays synthetic.
- **Pipeline:** orbit video → COLMAP poses (`pycolmap`) → Gaussian splat (`gsplat`) → metric alignment with a ChArUco board → collision mesh (splat → mesh → convex pieces) → attach to the LIBERO bodies.
- **Remaining shift:** table, background, lighting and camera come from LIBERO.

### Solution 3: scan the whole scene and add it to the simulator

Reconstruct the full table scene as a splat. Render the simulator from the real camera viewpoint. Composite the robot into the image by depth.

- **Distribution shift:** the smallest visual gap of the four. The rendered view is similar to the real view.
- **Cost:** it needs a good scan, careful metric and frame alignment, GPU time for splat training, and compositing code. The physics still come from LIBERO. A rendered scan is only a picture. It does not make the dynamics equal to the real dynamics.

### Solution 4: use real objects with the same size as the LIBERO objects (simplest)

Select ready LIBERO objects (bowl, plate). Find real objects with the same size. Use them for the hand data collection.

- **How:** read the bounding box of each LIBERO object asset. Buy or find real objects that match the width, depth and height. Put the markers on them. Record the hand demos as usual.
- **Advantage:** no scanning, no reconstruction and no extra GPU work. The grasp height, the aperture and the carry path from the human demo are correct for the simulated object.
- **Disadvantage:** there is a distribution shift between the objects and the scene. The real objects match the size, but not the texture, color or material. The real table, the background and the lighting do not match LIBERO. This shift is small for a motion-only pipeline, because the policy does not see real pixels. It is large for any approach that uses real pixels.

### What this project does

| Step | Approach | Status |
|---|---|---|
| Baseline | **Motion-only retargeting with size-matched objects (Solution 4).** The video is a motion-capture device. Only a few object-relative numbers cross from real to sim: approach direction, grasp height, lift, carry path, place offset, release timing. SmolVLA does not see a real pixel. | core path |
| Main visual | **SceneTwin:** copy the real object layout and camera pose into the simulator. Replay. Render from the same viewpoint next to the real clip. | built; one real clip shown next to its replay (the replay timed out) |
| Stretch | **Solutions 2 and 3 with Gaussian splats:** attach scanned objects (and scene) to the LIBERO bodies. Composite with the mesh robot by depth. | pipeline built, checked on stand-in scans and two real phone clips of one bowl (COLMAP's defaults registered 39 of 120 frames of one clip, so SfM uses a lowered SIFT threshold; see the [design log](design_decisions.md)), time-gated |

The baseline avoids the visual domain gap. It does not close the gap. Splat-based real2sim reduces the visual gap for illustration. A real-robot validation is outside this challenge.
