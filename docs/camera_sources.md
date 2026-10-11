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

## No printer: show the markers on a tablet

```bash
uv run python scripts/01_calibrate_camera.py generate-screen-markers \
    [--background table_photo.jpg] [--ppi 217] [--screen 1800x1200]   # -> data/markers/screen_markers.png
```

It renders the table (10 cm), bowl and plate (5 cm) markers at true size for the display's PPI (default: Surface
Go 1, 1800x1200, 217 PPI), optionally over a photo of your tabletop. Show the PNG fullscreen at 100% (1 image pixel per
screen pixel; set Windows display scaling accordingly if the viewer rescales), brightness high, and **measure the table
marker's black square with a ruler: 100 mm**. If it is off, rescale `--ppi` by measured/100 and regenerate. The metric
scale of everything comes from that size.

**Does the tablet thickness matter?** Only for absolute height. The marker plane is the glass, so table-frame z = 0 is
8.3 mm (Surface Go 1) above the real table; x/y are unaffected because the tablet is flat. In the default delta mode
nothing changes. With `--table-height`, pass `--marker-height-m 0.0083` so a hand resting on the table reads 0 above the
sim table. The default is 0. Also matte/glossy glare and the screen's refresh flicker can hurt detection: lower the
exposure or tilt the camera if the marker drops out.

## Using a phone as the camera

Tested on a Galaxy S24 (SM-S921B, Android 16) with option **B** (scrcpy over USB): it streamed to `/dev/video10`,
the ChArUco calibration ran through it (1920x1080, fx about 1311 px) and hand teleoperation worked. Options A, C and D
are documented recipes that were not run on a phone; the V4L2, network-stream and file paths were tested with the
laptop camera, a local MJPEG server and a clip. Ordered by how little they need:

### A. USB webcam mode (no software)

Android 14 QPR1 and later can expose the phone as a standard UVC webcam ([AOSP docs](https://source.android.com/docs/core/camera/webcam)):
plug in a USB-C **data** cable, then either pull down the notification shade and tap the USB notification, choosing
*Webcam*, or **Settings, USB preferences, Webcam**; `adb shell svc usb setFunctions uvc` does the same. It then appears
as `/dev/videoN` (check with `scripts/13_teleop.py sources`), and `hand /dev/videoN` works. Only MJPEG/YUYV are offered.
I could not confirm that Samsung's One UI ships this switch on the S24 (Samsung's own "camera share" is wireless and
aimed at Galaxy Books), so look for the option before relying on it; if it is missing, use B.

### B. scrcpy camera mirroring to a virtual webcam (USB, no phone app)

Needs Android 12+ on the phone and **scrcpy 2.2 or newer** on the PC (checked with 5.0.1). Ubuntu 22.04's apt package
is 1.21 (too old) and the snap is a third-party upload, so take the static build from the
[official releases](https://github.com/Genymobile/scrcpy/releases); it bundles `adb`. Camera options:
[scrcpy camera docs](https://github.com/Genymobile/scrcpy/blob/master/doc/camera.md).

**Installation (one time)**

1. scrcpy and its bundled adb, into your home directory (no root needed):

   ```bash
   mkdir -p ~/tools/scrcpy && cd ~/tools/scrcpy
   curl -sLO https://github.com/Genymobile/scrcpy/releases/download/v5.0.1/scrcpy-linux-x86_64-v5.0.1.tar.gz
   tar xzf scrcpy-linux-x86_64-v5.0.1.tar.gz --strip-components=1 && rm scrcpy-linux-x86_64-v5.0.1.tar.gz
   echo 'export PATH="$HOME/tools/scrcpy:$PATH"' >> ~/.bashrc && source ~/.bashrc
   scrcpy --version && adb version          # scrcpy 5.0.1 / Android Debug Bridge 1.0.41
   ```

   Use this adb, not `sudo apt install adb`: two different adb versions keep restarting each other's server.
   Newer releases: <https://github.com/Genymobile/scrcpy/releases>.

2. The virtual webcam (v4l2loopback), which scrcpy writes into and `13_teleop.py` reads from:

   ```bash
   sudo apt install v4l2loopback-utils v4l2loopback-dkms   # the dkms module builds for the running kernel
   sudo modprobe v4l2loopback video_nr=10 card_label=PhoneCam exclusive_caps=1
   ls -l /dev/video10 && v4l2-ctl --list-devices           # "PhoneCam" must be listed
   ```

   The `modprobe` has to be repeated after every reboot. To load it at boot instead:

   ```bash
   echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
   echo 'options v4l2loopback video_nr=10 card_label=PhoneCam exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
   ```

   `exclusive_caps=1` is needed so the device shows up as a capture camera (otherwise OpenCV cannot open it).

3. The project environment (nothing phone-specific): `uv sync`, then
   `uv run python scripts/13_teleop.py sources` lists `/dev/video10` once scrcpy is streaming.

4. On the phone: Settings, About phone, Software information, tap *Build number* 7 times; then
   Developer options, turn on **USB debugging** (a separate switch from developer mode itself). Android 12 or newer.

**Each session**

```bash
sudo modprobe v4l2loopback video_nr=10 card_label=PhoneCam exclusive_caps=1   # skip if /dev/video10 exists
adb devices          # USB-C data cable, phone unlocked, accept the key prompt; must say "device"
scrcpy --list-cameras                      # note the id of the main back camera (1x, "back")
scrcpy --video-source=camera --camera-id=0 --camera-size=1920x1080 --camera-fps=30 \
       --no-audio --v4l2-sink=/dev/video10 --no-playback --stay-awake
```

Leave that terminal running (`v4l2 sink started to device: /dev/video10` means it works) and use `/dev/video10` as
the source in a second terminal:

```bash
uv run python scripts/01_calibrate_camera.py capture --source /dev/video10   # once per camera and mode
MUJOCO_GL=egl uv run python scripts/13_teleop.py hand /dev/video10 --resolution 1920x1080 --fullscreen
```

Rules that avoid the usual failures:

- **Stop scrcpy with Ctrl+C**, never by unplugging. An abrupt disconnect leaves the camera session open on the phone.
- Keep the phone unlocked and awake (`--stay-awake`, plus a long screen timeout); Android drops the camera when the
  screen locks. Do not open other camera apps meanwhile.
- Use a short data cable straight into a PC port, no hub.
- The `--camera-size` here and the `--resolution` of `13_teleop.py` must match the size the intrinsics were
  calibrated at, or the script refuses to start.
- With markers the image must **not** be mirrored (ArUco cannot read mirrored markers). `13_teleop.py` only mirrors
  webcams with `--no-markers`, and the live view says `image is MIRRORED` if it detects the problem.

**Troubleshooting**

| Symptom | Cause and fix |
|---|---|
| `adb devices` is empty, `lsusb` shows the phone (Samsung `04e8:6860`, MTP) | USB debugging is off or its interface is not active. Toggle USB debugging off and on, *Revoke USB debugging authorizations*, replug with the screen unlocked and tap Allow, set the USB notification to *File transfer*. One UI 6+: also turn **Auto Blocker** off. Check with `lsusb -v -d 04e8:6860 \| grep -E "bInterface(Class\|SubClass\|Protocol)"`: class 255, subclass 66, protocol 1 is adb. |
| `adb devices` says `unauthorized` | Accept the RSA prompt on the phone (tick "Always allow"); if it never appears, revoke authorizations and replug. |
| `adb: command not found` | The bundled adb is only in `~/tools/scrcpy`; export the PATH line above (or open a new shell). |
| `ERROR: Could not find any ADB device` | Same as the empty `adb devices` row. |
| `Camera disconnected`, then `Device disconnected` | The phone locked, another app took the camera, or the cable dropped. Use `--stay-awake` and a better cable. |
| `The system-wide limit for number of open cameras has been reached` | A stuck session from a crashed run still holds the camera. Unplug, wait 10 s, `adb kill-server`, replug; if it persists `adb shell pkill -f com.genymobile.scrcpy.Server`, then restart the phone. Inspect with `adb shell dumpsys media.camera`. |
| scrcpy runs but the teleop window says `table marker: NOT FOUND` | Wrong id in view (`seen ids` lists them), marker too small or glared, or `image is MIRRORED`. |

Phones older than Android 12 (a Galaxy S10e only reaches Android 12 on its final update) cannot use
`--video-source=camera`; update them or use C. Calibrate every phone separately: intrinsics are per camera and mode.

Latency is the lowest of the options (a direct H.264 stream over USB). Declared camera sizes/fps are not always
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

Use B: lowest latency, nothing installed on the phone, and the one that is verified. A is a one-minute try if the
phone offers a *Webcam* USB mode. Use C for a quick look from the sofa. Whichever you pick: lock the phone's autofocus/exposure if the app allows, use the main (1x) back
camera, mount it on the tripod, and **calibrate through the same source** before trusting scale.
