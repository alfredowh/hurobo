# Camera sources for live hand teleoperation

`scripts/13_teleop.py hand SOURCE` takes the human side from any of these:

| SOURCE | Meaning | Pipeline |
|---|---|---|
| `data/raw/.../hand_moves_cube` (folder) | stand-in frames, markers pasted in | offline replay, slow-motion + videos |
| `clip.mp4` | real recording with printed markers | offline replay (needs `--intrinsics`) |
| `webcam`, `webcam:2`, `2`, `/dev/video2` | a V4L2 device: laptop webcam, UVC phone, scrcpy / DroidCam bridge | live |
| `http://…`, `https://…`, `rtsp://…` | network stream, e.g. a phone IP-camera app | live |
| `file:clip.mp4` | a video played through the live pipeline, for trying it without hardware | live |

`uv run python scripts/13_teleop.py sources` lists the V4L2 devices the kernel knows about. This laptop has
`/dev/video0` (HP 5MP RGB camera) and `/dev/video2` (its IR camera: do not use); the odd-numbered nodes are metadata, not pictures.

## Quickest test: built-in webcam, no markers, relative motion only

```bash
MUJOCO_GL=egl uv run python scripts/13_teleop.py hand webcam:0 --no-markers
```

Nothing to print or calibrate. The first frame with a hand in view anchors the robot: that hand position is the
robot's current position and everything after is a delta. Then:

- hand right / up / **away from the camera** moves the gripper right / up / away from the agentview camera, as you
  see it in the agentview panel (the webcam image is mirrored by default, like a video call);
- pinch thumb and index to close the gripper (the cube appears on the closing edge);
- `r` re-anchors (clutch), `esc`/`q` quits; `--scale 1.5` makes the robot move more per centimetre of hand.

What this mode assumes, and what it gives up:

- the camera stays still and is roughly **level** (`--camera-pitch 15` if it looks 15 degrees down; a laptop screen
  tilted back is a small negative angle). Roll is assumed zero, so "up" is always world up;
- the camera is not tied to a table, so there is no table-frame trajectory, no `--table-height`, and the
  reprojected robot ring in the camera panel is measured from the first-hand anchor only;
- distance from the camera comes from MediaPipe's hand-size prior and guessed intrinsics, so depth (robot x) is the
  noisiest axis and its scale is only approximate. Left/right and up/down are far better.

## What a live camera needs with markers

1. **A static camera** (tripod). The table pose is averaged over the first `--calib-frames` detections and then locked;
   if you move the camera, restart.
2. **The table marker in view**: ArUco `DICT_4X4_50`, id 20, the *black square* printed at exactly 10.0 cm.
   `uv run python scripts/01_calibrate_camera.py generate-markers` writes printable PNGs to `data/markers/`.
   The marker's x/y axes become the table frame the hand moves in; its size sets the metric scale.
3. **Intrinsics for that camera in that mode.** Without `data/camera_intrinsics.npz` the script guesses
   fx = 0.8 x width and warns; hand scale and depth are then only roughly right.
   Calibrate with the same source: `uv run python scripts/01_calibrate_camera.py capture --source webcam:0`
   (show the ChArUco board from `generate-board`; `c` captures, `q` finishes). Resolution, zoom and lens (main vs
   ultrawide) must be the ones you will teleoperate with.
4. **Good hand visibility.** Pinch with thumb and index finger in view. Depth from a single camera is the weakest
   axis (the hand-size prior); use `--scale` and the clutch (`r`) rather than expecting millimetres.

In the window: `r` re-anchors (move your hand somewhere comfortable, press `r`, the robot's current pose becomes the new
reference), `esc`/`q` quits. Optional: `--record` writes `data/teleop/live/teleop_live.mp4`, `--table-height` grounds
height to the sim table, `--align agentview|table|<deg>` picks how the table frame is rotated into the sim.

## Using the Galaxy S24 as the camera

Nothing here has been run against the S24 itself (no phone was connected when this was written). What *was* tested:
the V4L2 webcam path on this laptop's camera, the network-stream path against a local MJPEG server, and the file path.
So the commands below are the documented recipes, ordered by how little they need:

### A. USB webcam mode (no software)

Android 14 QPR1 and later can expose the phone as a standard UVC webcam ([AOSP docs](https://source.android.com/docs/core/camera/webcam)):
plug in a USB-C **data** cable, then either pull down the notification shade and tap the USB notification, choosing
*Webcam*, or **Settings, USB preferences, Webcam**; `adb shell svc usb setFunctions uvc` does the same. It then appears
as `/dev/videoN` (check with `scripts/13_teleop.py sources`), and `hand /dev/videoN` works. Only MJPEG/YUYV are offered.
I could not confirm that Samsung's One UI ships this switch on the S24 (Samsung's own "camera share" is wireless and
aimed at Galaxy Books), so look for the option before relying on it; if it is missing, use B.

### B. scrcpy camera mirroring to a virtual webcam (USB, no phone app)

Needs Android 12+ and **scrcpy 2.2 or newer**. Ubuntu 22.04's apt package is 1.21 (too old) and the snap is a
third-party upload, so take the static build from the [official releases](https://github.com/Genymobile/scrcpy/releases)
(it bundles `adb`). Camera options: [scrcpy camera docs](https://github.com/Genymobile/scrcpy/blob/master/doc/camera.md).

```bash
sudo apt install v4l2loopback-utils        # the module for kernel 6.8.0-138 is already installed
sudo modprobe v4l2loopback video_nr=10 card_label=PhoneCam exclusive_caps=1
# phone: Settings > About > tap Build number 7x; Developer options > USB debugging; accept the PC's key
adb devices                                # should list the phone as "device"
scrcpy --list-cameras                      # note the id of the main back camera
scrcpy --video-source=camera --camera-id=0 --camera-size=1920x1080 --camera-fps=30 \
       --no-audio --v4l2-sink=/dev/video10 --no-playback        # --no-mirror on older versions
uv run python scripts/13_teleop.py hand /dev/video10 --resolution 1920x1080
```

Latency is the lowest of the options (it is a direct H.264 stream over USB). Declared camera sizes/fps are not always
accurate; check what you get with `v4l2-ctl -d /dev/video10 --list-formats-ext`.

### C. Wi-Fi: phone IP-camera app (no PC-side setup)

Run an IP-camera app on the phone (e.g. "IP Webcam"), same Wi-Fi as the PC, and pass its URL:

```bash
uv run python scripts/13_teleop.py hand http://PHONE_IP:8080/video --resolution 1280x720
```

MJPEG over HTTP and RTSP both work (`CaptureFrameSource` keeps only the newest frame on a thread and requests
low-latency FFmpeg options, but RTSP typically still adds 1 to 2 s and MJPEG a few hundred ms, which you will feel).
Set the app to 1280x720 and fix focus/exposure.

### D. DroidCam

Phone app plus the [Linux client](https://github.com/dev47apps/droidcam-linux-client): `droidcam-cli adb 4747` over
USB (or `droidcam-cli PHONE_IP 4747` over Wi-Fi) feeds a v4l2loopback device, used like B. More moving parts than B;
only worth it if scrcpy's camera mode is not available.

### Recommendation

Try A (one minute). If the option is missing, use B: lowest latency, nothing installed on the phone. Use C for a quick
look from the sofa. Whichever you pick: lock the phone's autofocus/exposure if the app allows, use the main (1x) back
camera, mount it on the tripod, and **calibrate through the same source** before trusting scale.
