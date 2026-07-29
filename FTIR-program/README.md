# USB Camera Detector

Started as a program for a USB-connected camera plugged directly into
this PC, frames coming straight from `cv2.VideoCapture` -- deliberately
kept separate from a standalone VDO.Ninja Playwright prototype one level
up (`../test_code.py`, since removed from that location) for a long
time. That's no longer true: this app now supports VDO.Ninja as a
second, alternate camera source (see "Camera sources and calibration
profiles" below), adapting that prototype's direct-video-element capture
technique into its own module (`vdo_ninja_source.py`) rather than
importing or wrapping the standalone script.

This is being built as a staged pipeline toward quantitative bright-spot
measurement (contact area, brightness) for FTIR data collection. Full
original plan: 10 stages from camera connection through lens distortion
correction, physical/brightness calibration, multi-level blob detection,
per-circle assignment, and measurement/logging. Each stage below is
meant to be independently testable before the next begins -- though the
actual end state diverged from "per-circle assignment": per-pad
measurement ended up built around user-drawn ROI rectangles on recorded
video instead (see "Recording workflow" further down), not circle
grouping on the live feed.

**Stage 1 (done): reliable camera connection and live preview.**
No blob/bright-spot detection, no calibration, no measurement UI --
that's later stages, in `processing.py`, once this stage is solid.

**Stage 2 (done): camera settings inspection and locking.**
Exposure/gain must be fixed (not auto) before any brightness measurement
means anything. See "Camera settings (Stage 2)" below.

**App shell (done): tabbed desktop application (`gui_app.py`).**
Live Camera, Settings, Calibration, Detection, Recordings, Processing,
Results. Wraps the Stage 1/2 code as-is (see "Desktop app" below). The
last three tabs (added well after the original 10-stage plan below --
see "Recording workflow" further down) are where this project actually
ended up: record lossless footage once, then iterate on detection tuning
and per-pad measurement against the same saved frames as many times as
needed, instead of only ever detecting live.

**Stage 3 (done): lens distortion correction (ChArUco calibration).**
Board parameters are input fields, not hardcoded -- see "Calibration tab
(Stage 3)" below.

**Stage 4 (done): physical scale calibration (mm-per-pixel).**
Two-point-click + known-distance, on the undistorted frame -- see
"Physical scale calibration (Stage 4)" below.

**Stages 5-6, live (done): continuous, background-subtracted, arbitrary-
shaped fragment detection.** Detects live off the shared camera stream
(no second connection), against a saved background reference rather than
raw brightness, with no blur/morphological closing and no assumption
that a detected region is circular. See "Detection tab (Stages 5-6 live
rebuild, plus part of 8)" below. Fragment-to-circle grouping/area
coverage (Stage 7 + part of 8) was built (`assignment.py`/
`measurement.py`) but never wired into the live view, and was later
removed once the Processing tab's per-pad ROI analysis replaced the
whole approach (see "Recording workflow" further down). Normalized
0-100% brightness (Stage 9) isn't built yet -- that needs a separate
dark-reference/max-reference calibration (distinct from the background
reference above, which only cancels static structure, not a brightness
scale) -- not built.

**Camera sources + VDO.Ninja (done): a second, alternate camera source,
completely isolated calibration per source, and a color-based detector
that swaps in automatically for it.** See "Camera sources and
calibration profiles" below. This stage was about making both sources,
their separate calibration profiles, and detector switching work
reliably first; per-pad measurement came later and took a different
direction (per-ROI, not circle grouping -- see "Recording workflow"
further down).

Camera: InnoMaker/UVC industrial camera, model **U20CAM-9281M**. The
log2-seconds exposure math and `CAP_PROP_GAIN`/`CAP_PROP_AUTO_EXPOSURE`
calls in `settings.py` are ported from a vendor reference control script
we have on hand. Trigger mode (a best-effort toggle via
`CAP_PROP_AUTOFOCUS`) is documented in that same script but not
implemented here -- out of scope until a later stage needs it.

## Install

You need Python with a few packages:

```
pip install opencv-python numpy pygrabber comtypes PySide6 playwright
playwright install chromium
```

`PySide6` is only needed for the desktop app (`gui_app.py`) -- the CLI
tools (`preview.py`, `settings.py`) don't use it. `playwright` (plus its
`chromium` browser download) is only needed if you're going to use the
VDO.Ninja camera source -- the USB camera path never imports it.

`opencv-python`/`numpy` do the actual capture. `pygrabber`/`comtypes` are
used only for read-only diagnostics (`--list` device names, `--list-formats`)
-- they talk to the same DirectShow `IAMStreamConfig` API that AMCap's
format dialog uses, which is how we know the real (resolution, pixel
format, fps range) combinations a camera supports instead of guessing.
Capture itself still goes through plain `cv2.VideoCapture`; if pygrabber
isn't installed or fails on your machine, capture still works, you just
lose the friendly device names and `--list-formats`.

(If you already ran the VDO.Ninja program in the parent folder,
`opencv-python`/`numpy` are already installed system-wide.)

## Files

- `camera.py` -- device probing, a `CameraStream` wrapper around
  `cv2.VideoCapture`, and pygrabber-based capability enumeration. No
  display/GUI code here.
- `settings.py` (Stage 2) -- camera exposure/gain/auto-exposure inspection
  and locking. Independent of `camera.py`/`preview.py`; takes the raw
  `cv2.VideoCapture` from a `CameraStream` (via its public `.cap`
  attribute) and can be run standalone or imported.
- `processing.py` -- currently just a passthrough `process_frame()`. This
  is where bright-spot detection will be added in a later stage, without
  needing to touch capture or preview code.
- `preview.py` -- Stage 1 entry point: the live preview loop and CLI.
  Optionally reports (read-only) Stage 2 settings via `--report-settings`.
  Still works standalone -- the desktop app doesn't replace it.
- `config_store.py` -- the data-storage module: small JSON load/save
  helpers, currently used for Settings tab persistence, meant to hold
  calibration data too once that stage exists.
- `camera_controls.py` -- brightness/contrast/saturation/hue/gamma/
  sharpness/backlight/gain/white-balance get/set via `cv2.VideoCapture`,
  with the same verify-and-retry pattern as `settings.py`'s exposure/gain
  functions. Independent of `settings.py` (not modified to add this).
- `distortion.py` (Stage 3) -- ChArUco board calibration
  (`CharucoCalibrator`: detect/capture/calibrate), `undistort_with_maps()`,
  and `crop_edges()`. `save_calibration()`/`load_calibration()` both take
  an optional `path=` (defaulting to the flat `calibration_data.json`) --
  this is what let per-source calibration profiles (below) exist without
  changing anything in this file: every caller just passes a
  profile-specific path instead of relying on the default.
- `scale.py` (Stage 4) -- `compute_scale()` (two points + a known real
  distance -> mm-per-pixel) and persistence, same `path=` pattern as
  `distortion.py`. Independent of `distortion.py` (separate storage, per
  the project's module-separation pattern), though the Calibration tab UI
  requires a distortion calibration to already exist before using it.
- `circle_config.py` (Stage 5) -- the USB detector's three difference
  thresholds, and (additive) the VDO.Ninja color detector's own settings
  (color preset, hue/sat/val thresholds, its own three confidence
  thresholds, mask visibility, processing FPS) -- one shared schema, same
  `path=` pattern. (Also used, unmodified, by the Processing tab's
  offline detection -- see "Recording workflow" below. Originally also
  held expected-circle-count/area/tolerance fields; removed along with
  `assignment.py`/`measurement.py` below, once the Processing tab's
  per-pad ROI analysis replaced circle-grouping entirely. The filename
  predates that removal.)
- `background_reference.py` -- `save_background_reference()`/
  `load_background_reference()` (a corrected frame, captured with no
  intended contact light) and `compute_difference()` (nonnegative
  current-minus-background). USB-only -- used by the live Detection tab
  so static structure (fixture reflections, LED hot spots) never gets
  mistaken for real contact light. Same `path=` pattern.
- `detection.py` (Stage 6) -- `detect_fragments()`: multi-level fragment
  detection on any single-channel array (raw grayscale difference, or a
  color-confidence map), no blur, no morphological closing, no assumption
  a fragment is circular. Shared verbatim by both detectors -- see
  "Camera sources and calibration profiles" below for how
  `color_detection.py` reuses it unchanged.
- `assignment.py` / `measurement.py` (Stages 7-8) -- **removed.** Grouped
  detected fragments into physical circles by location/distance/overlap
  with a computed expected-circle ROI (`assign_fragments_to_circles()`),
  then measured per-circle possible/probable/strong area and coverage
  (`measure_circles()`). Tested and correct at the time, but never called
  by the live Detection tab, and the Processing tab's later per-pad ROI
  analysis (user-drawn rectangular search regions, independent detection
  per region -- see "Recording workflow" below) replaced the whole
  approach rather than ever wiring these in. Confirmed zero references
  anywhere before removal.
- `calibration_profiles.py` -- the profile registry: per-camera-source
  (`"usb"` / `"vdo_ninja"`) named profiles, each with its own subfolder
  (`profiles/<id>/`) holding that profile's own `calibration_data.json`/
  `scale_calibration.json`/`circle_config.json`/`background_reference.npy`
  -- via the `path=` pattern above, not by changing any of those modules.
  Also the one-time migration of the original flat files into a default
  "USB Camera" profile. See "Camera sources and calibration profiles".
- `vdo_ninja_source.py` -- Playwright-based VDO.Ninja viewer capture,
  adapted from `../test_code.py`'s direct-video-element technique,
  restructured as a start/stop object (`open()`/`read()`/`get_info()`/
  `release()`) duck-typing `camera.CameraStream`'s shape. All
  Playwright/browser code lives here and only here.
- `color_detection.py` -- the VDO.Ninja color detector:
  `build_confidence_map()` turns an HSV frame into a single-channel
  "color confidence" array (mirroring what `background_reference.py`
  does for grayscale), then `detect_color_fragments()` feeds it into
  `detection.detect_fragments()` **unchanged** and filters for hysteresis
  (a region survives only if it reaches the "core" confidence tier
  somewhere in it).
- `regions.py` -- `DetectedRegion`: the common per-region structure both
  detectors' output gets wrapped into (adds `source_type` and mm² areas
  to a `detection.Fragment`), so the Detection tab's reporting code is
  written once, not duplicated per detector.
- `camera_identity.py` -- stable USB camera identification via
  DirectShow's `DevicePath` property (survives reboots/replugging, unlike
  OpenCV's index), with a friendly-name fallback for the rare driver that
  doesn't expose one. See "Two USB cameras" below.
- `fisheye_calibration.py` -- **removed.** Was a fisheye (Kannala-Brandt)
  ChArUco calibrator, API-parallel to `distortion.py`'s pinhole
  `CharucoCalibrator`, built and tested against synthetic fisheye data
  but never wired into any tab (both camera roles use the pinhole model
  -- see "Two USB cameras" below). Confirmed zero references anywhere
  before removal; revive from git history if edge-accurate wide-angle
  correction is ever actually needed.
- `gui_app.py` -- desktop app entry point (`MainWindow`, tab wiring, the
  dockable controls panel, `latest_frame` sharing, `active_source_type` /
  `active_profile_key`, `frame_dispatcher` -- see "Recording workflow"
  below).
- `tabs/` -- one file per tab (`live_camera_tab.py`, `settings_tab.py`,
  `calibration_tab.py`, `detection_tab.py`, `recordings_tab.py`,
  `processing_tab.py`, `results_tab.py`), plus `camera_controls_panel.py`
  (the reusable sliders widget embedded in the Settings tab and
  instantiated again for the dockable panel). Each is UI only -- they
  call into `camera.py`/`settings.py`/`camera_controls.py`/
  `distortion.py`/`config_store.py`/`calibration_profiles.py`/
  `vdo_ninja_source.py`/`recording_store.py`/`processing_project.py`/
  `detection_pipeline.py` rather than reimplementing any of that logic.

## Find your camera index

Plain OpenCV has no API to list camera names -- "detecting" a camera the
cv2 way means trying sequential indices (0, 1, 2, ...) and seeing what
responds. `--list` now also cross-references pygrabber for the real
DirectShow friendly name of each index (e.g. "USB Camera"), when available:

```
py preview.py --list
```

This prints a table of which indices opened, whether they delivered a
frame, at what resolution/backend, and their name. If you have a built-in
laptop webcam as well as the USB camera, the name column should make it
obvious which is which without guessing from resolution alone.

Each index is probed with a timeout (default 3s, adjust with
`--probe-timeout`) so a bad or busy index can't hang the whole scan.

## See what your camera actually supports

Before picking resolution/fps/format values, see the real capability list
the driver advertises (the same data AMCap's format dialog is built on):

```
py preview.py --index 1 --list-formats
```

This is the fastest way to answer "why can't I get fps X at resolution Y" --
different pixel formats at the *same* resolution can have wildly different
fps ranges (see "About the 120fps spec" below for a concrete example).

## Run the live preview

With a known index (defaults to the U20CAM-9281M's native 1280x800 @
120fps, requesting MJPG -- see below for why that combination matters):

```
py preview.py --index 1
```

Without `--index`, it probes automatically and asks you to pick:

```
py preview.py
```

Override resolution/fps/format:

```
py preview.py --index 1 --width 640 --height 480 --fps 60 --fourcc MJPG
```

- `--backend {any,dshow,msmf}` -- OpenCV capture backend, **defaults to
  `dshow`**. This is a deliberate choice for the U20CAM-9281M: the
  vendor's own Windows control script explicitly avoids MSMF ("MSMF
  (default) backend has poor UVC control support") and hardcodes DSHOW.
  Try `any` or `msmf` only if `dshow` fails to open the camera.
- `--width` / `--height` / `--fps` -- requested capture settings, default
  to `1280` / `800` / `120`. The driver is not guaranteed to honor these --
  the program always prints what was actually granted (see below), and the
  on-screen preview shows *measured* fps, not the request.
- `--save-dir` -- where frames saved with the `s` key go (default
  `captures/`, created automatically).

### About the 120fps spec (now largely solved)

This camera is spec'd for up to 120fps at 1280x800. Requesting `--fps 120`
alone used to measure only ~10fps in practice -- confirmed why with
`--list-formats`: at 1280x800, **YUY2** (the driver's default pixel format
when nothing else is requested) is hard-capped at exactly 10fps by the
driver itself; **MJPG** at the identical resolution supports the full
10-120fps range. This isn't a bandwidth estimate, it's what the driver
itself advertises via `IAMStreamConfig` -- the same data AMCap's format
dialog reads.

Fixing this needed two things, both now built into `CameraStream.open()`:

1. Actually request `--fourcc MJPG` (now the default) instead of leaving
   the format unset.
2. **Property-set order matters and is easy to get backwards**: FourCC
   must be the *last* `cv2.VideoCapture.set()` call, after width/height/fps
   -- setting it earlier gets silently reverted to YUY2 by the later calls.
   This isn't documented anywhere obvious; found by testing every ordering
   directly against hardware.

With both fixed, measured throughput on the reference test device went
from ~9fps to ~73fps at 1280x800 -- short of the full 120 spec (likely
exposure-related: very short exposure times are needed to sustain 120fps,
which is genuine Stage 2+ camera-control territory), but a 7-8x
improvement from just fixing format selection, with no resolution or fps
compromise. If your real camera still underperforms after this, run
`--list-formats` first to confirm what it actually supports before
assuming the code is at fault. Reaching the full 120 (vs. the ~73fps
confirmed above) is likely exposure-related -- see "Camera settings
(Stage 2)" below, now that exposure control is actually built.

## Camera settings (Stage 2)

Before any brightness measurement can be trusted, exposure and gain need
to be fixed at known values, not left on auto -- auto-exposure will
silently re-brighten or darken the image to compensate for whatever's in
frame, which would invalidate every brightness comparison a later stage
makes. `settings.py` inspects and locks these, ported from the vendor's
U20CAM-9281M control script (DirectShow expresses exposure as
log2-seconds, not the UVC 100us units used on Linux/macOS).

Check current settings:

```
py settings.py --index 1
```

Lock manual exposure and/or gain:

```
py settings.py --index 1 --exposure-ms 5 --gain 32
```

Restore auto exposure:

```
py settings.py --index 1 --auto-exposure
```

These settings persist on the camera/driver between runs (confirmed
directly -- they're hardware/firmware-level controls, not tied to a
particular capture session), so you don't need to re-lock them every time
you run `preview.py`. Add `--report-settings` to `preview.py` to see the
current exposure/gain alongside the usual startup info (read-only there;
change them with `settings.py`).

**Known quirk, handled automatically**: the very first control-change
request after opening a device can be silently dropped by the driver,
with every identical request after that succeeding (observed directly
against real hardware). `settings.py` verifies the readback after every
set and retries automatically before reporting a real failure -- you
should only ever see a `WARNING:` line if a value genuinely isn't
supported (e.g. a gain outside the camera's range), not from this
first-attempt flakiness.

If `Auto-exposure mode` reports `unknown (raw value ...)` instead of
`manual`/`auto`, that means this particular device/driver doesn't
implement `CAP_PROP_AUTO_EXPOSURE` the way DirectShow's 0.25/0.75
convention expects -- worth confirming on the real U20CAM-9281M, since
the vendor's own script demonstrates it working on that exact model.

### Exposure is quantized -- your requested ms won't be exact

DirectShow only supports discrete log2-second exposure steps (500ms,
250ms, 125ms, ... 31.25ms, 15.6ms, 7.81ms, 3.91ms, 1.95ms, ...). Requesting
`--exposure-ms 5` will always round to the nearest real step (3.91ms here,
not 5.00ms) -- that's a hardware/driver limitation, not a bug. Check the
printed `exposure_ms` in the report to see what you actually got, and pick
a value close to a real step if you want to reason about it precisely.

### All other controls: the native property dialog

`settings.py` only handles exposure and gain programmatically, because
those two directly and multiplicatively scale measured brightness and are
the ones that need scriptable, verified, repeatable locking across
sessions. Everything else the driver exposes -- brightness, contrast,
hue, saturation, sharpness, gamma, white balance, backlight compensation,
focus, etc. -- is a "set once to a sane value, then leave it alone"
control, not something that needs re-locking every run.

For those, rather than reimplementing sliders (which AMCap doesn't do
either -- that dialog *is* the driver's own native property page, not a
custom UI), this calls it directly:

```
py settings.py --index 1 --configure
```

This blocks until you close the dialog, then prints the resulting
exposure/gain. It's the identical dialog AMCap shows (confirmed: this
camera reports 2 property pages, matching the "Video Proc Amp" / "Camera
Control" tabs in AMCap's window) -- same sliders, same ranges, guaranteed
complete, since it's driver-rendered rather than something we'd have to
keep in sync ourselves.

**For a measurement application, treat most Video Proc Amp controls as
things to *disable*, not tune**: gamma should stay linear (100), auto
white balance is meaningless for a monochrome sensor and worth turning
off in case it's still perturbing raw values, and brightness/contrast
"enhancement" generally works against getting raw linear sensor data.
Exposure and gain are the two that should be deliberately set (via
`settings.py` itself, not this dialog) and locked for the session.

If you'd rather have specific additional controls (e.g. focus) scriptable
and verified the same way exposure/gain are, say which ones matter for
your setup and they can be added the same way.

### One tool, live: settings + preview together

You don't need to run `settings.py --configure` and `preview.py`
separately -- press `c` in the preview window and the same native dialog
opens right there, *while the live preview keeps running*, so you see
slider changes take effect immediately instead of only after closing the
dialog.

This works because `OleCreatePropertyFrame` is a modal, blocking call --
if it ran on the same thread as the preview loop, the preview would freeze
while the dialog was open. Pressing `c` instead opens it on a background
thread (confirmed safe: pygrabber's separate camera binding coexists fine
with the already-open `cv2.VideoCapture`, verified directly), so the main
thread's `cv2.imshow`/`waitKey` loop -- and therefore the visible preview
-- never stops. Pressing `c` again while it's already open just prints a
notice rather than opening a second one. When you close it, the preview
prints the resulting exposure/gain so you can confirm what actually stuck.

## Desktop app

```
py gui_app.py
```

A tabbed window: Live Camera, Settings, Calibration, Detection,
Recordings, Processing, Results. This wraps `camera.py`/`settings.py` --
neither was modified to build it. The last three tabs are covered in
their own "Recording workflow" section further down, not here.

**How it's wired together** (why tab-switching can't restart or duplicate
the camera): `MainWindow` creates all five tab widgets once, up front, and
`QTabWidget` only hides/shows already-created widgets when you switch --
it never destroys and recreates them. The `CameraStream` itself is created
exactly once, by clicking Connect on the Live Camera tab, and stored on
`main_window.stream`; other tabs (Settings today) reach it through that
same reference rather than opening their own. Nothing about switching
tabs touches the stream at all. Confirmed directly: connected, switched to
Calibration and back, and checked the stream object was the identical
instance, not a new one.

- **Live Camera tab**: a dropdown of probed camera indices (with real
  device names, same as `preview.py --list`) populated on a background
  thread so probing's multi-second delay never freezes the window;
  Connect/Disconnect; live preview; a live-updating info line (backend,
  resolution, FourCC, measured fps). Frame reads happen on a `QTimer` on
  the main/GUI thread -- simple and safe (no cross-thread widget access)
  since each read only takes a few milliseconds at this camera's actual
  frame rate.
- **Settings tab**: embeds `CameraControlsPanel` (`tabs/camera_controls_panel.py`)
  -- every slider from the native "Video Proc Amp" dialog (Brightness,
  Contrast, Hue, Saturation, Sharpness, Gamma, White Balance + Auto,
  Backlight Compensation, Gain), plus Exposure/Auto Exposure using
  `settings.py`'s existing functions unchanged, plus Load/Save and
  "Open Native Dialog...". See "Camera controls panel" below for how the
  sliders are implemented and their real hardware ranges.
- **Calibration / Detection tabs**: placeholders at this point in the
  build -- lens distortion, blob detection, and measurement logic are
  all later stages, covered in their own sections below.

Closing the window releases the camera via `LiveCameraTab.stop()`,
whether or not you clicked Disconnect first.

**Verified directly** (offscreen Qt mode, against the real cameras in the
dev environment): app constructs with all 5 tabs; background probing
populates the dropdown with real device names; Connect opens a real
stream and the video label receives actual non-null frames with live
measured fps; switching tabs and back left the stream as the exact same
object; Settings save/load round-tripped correctly; closing the window
released the camera cleanly.

### Camera controls panel

Live Camera has a "Show Controls Panel" button that docks a *second*
instance of the same controls panel next to whatever tab is active --
including Live Camera itself, so you can watch the live image while
tweaking sliders, without needing to switch to the Settings tab (both
instances talk to the same `main_window.stream` independently; the
button and dock stay in sync if you close the dock's own close box
instead of clicking the button again).

**How the sliders are implemented** -- this took a real wrong turn worth
knowing about: the first approach queried the native `IAMVideoProcAmp` /
`IAMCameraControl` COM interfaces directly (defined by hand in a
since-removed module, the same way `pygrabber` defines its own DirectShow
interfaces), specifically to get exact hardware min/max/step ranges via
`GetRange()` -- the same data the native dialog itself uses, so slider
bounds would match it exactly. `QueryInterface` for both reliably failed
with `E_NOINTERFACE` in this environment, even on a device where `cv2`'s
own equivalent `CAP_PROP_BRIGHTNESS`/`CAP_PROP_CONTRAST` calls succeeded
moments earlier against the *same* filter -- confirmed the GUIDs were
correct (they parsed back to the exact expected strings) and that the
documented correct approach, `ICaptureGraphBuilder2::FindInterface`,
also failed (`E_FAIL`), most likely because it needs a connected/rendered
graph, which risks conflicting with the already-open capture stream.

Given `cv2.VideoCapture.get/set()` already reached every needed property
successfully and reliably (confirmed directly against both cameras in
this environment, with real verified value changes for brightness,
contrast, saturation, hue, gamma, sharpness, backlight, gain, and white
balance temperature), `camera_controls.py` uses that instead -- the same
mechanism `settings.py` already used successfully for exposure/gain, now
extended to the rest. The tradeoff: OpenCV doesn't expose real
`GetRange()` bounds, so slider min/max values in `camera_controls.py`
are reasonable generic defaults, not exact per-camera hardware limits.
Every set is verified via readback with the same retry-on-mismatch
pattern as `settings.py`'s exposure/gain functions, so a slider that hits
a request outside the camera's real range will show the driver's
actual clamped value, not a false confirmation.

Two controls from the native dialog are not included: **ColorEnable**
(greyed out/inapplicable on this monochrome sensor) and **PowerLine
Frequency** (no `cv2` property exists for it in this OpenCV build, and
it isn't part of the standard `IAMVideoProcAmp`/`IAMCameraControl`
property set either, so it would need the same COM route that proved
unreliable here).

**Reset to Defaults**: `LiveCameraTab` captures every control's value
(and the exposure/gain state) immediately after `stream.open()` succeeds,
before anything else touches them, and stores it on
`main_window.default_controls`/`default_exposure_info`. "Reset to
Defaults" restores those captured as-connected values -- this camera's
real power-on state, not a hardcoded guess, since OpenCV doesn't expose
true hardware defaults any other way. It's captured automatically the
next time you connect, so it works the first time even if the camera
isn't plugged in right now. Verified directly: changed brightness/
contrast/gamma/auto-white-balance away from their captured values,
clicked Reset, and confirmed via direct hardware readback (not just the
UI) that all four were restored correctly, with the sliders updating to
match.

## Camera sources and calibration profiles

Live Camera's "Source:" dropdown picks between "USB Camera" (unchanged
behavior from every earlier stage) and "VDO.Ninja Stream" (a phone/
browser camera reached over a WebRTC viewer link). Everything below
explains how the two coexist without one interfering with the other.

**Never two streams at once**: `main_window.stream` holds either a
`camera.CameraStream` or a `vdo_ninja_source.VdoNinjaSource` -- never
both. Switching the source-type combo (`LiveCameraTab._on_source_type_changed`)
always calls `.release()` on whatever was connected under the *old*
source type, synchronously, before the new source's controls even
become active -- there is no path where both could be alive
simultaneously, confirmed directly with fake stream objects standing in
for real hardware/browser connections (asserted `.release()` was called
and `main_window.stream` was `None` immediately after the switch, in
both directions).

**One shared interface, no new abstract base class**: `CameraStream`
already exposes exactly `.read()`/`.get_info()`/`.release()`, so
`VdoNinjaSource` was written to duck-type the same three methods (plus
its own `.open(url, fps)`) rather than inventing a formal interface --
nothing else in this codebase uses one, so this stays consistent. One
extra duck-typed attribute matters: `VdoNinjaSource.cap = None`, purely
so `CameraControlsPanel`'s existing `stream.cap is None` guards (every
exposure/gain/video-proc-amp method checks this before touching
hardware) already report "not connected" for a VDO.Ninja source with
**zero changes** to that already-working file -- this is how "hide/
disable USB-only controls" for VDO.Ninja is satisfied without a single
new `if source_type == ...` branch inside Settings/CameraControlsPanel.

**VDO.Ninja capture** (`vdo_ninja_source.py`): adapted from
`../test_code.py`'s already-proven technique -- Playwright launches
Chromium (always headless, no visible browser window), waits for a
`<video>` element to actually have live frames, binds a reusable
`<canvas>` to it, and each tick redraws the video frame onto that canvas
and reads it back as a PNG -- capturing the stream's true native
resolution directly, never a full-page screenshot (that file's own
docstring already covers why: screenshot-based and CDP screencast
approaches were both tried and were worse/broken). Restructured here as
a start/stop object instead of a blocking CLI loop: `open()` starts one
dedicated background thread that owns the *entire* Playwright lifecycle
(navigate, wait, capture loop, close) -- Playwright's sync API isn't
safe to touch from more than one thread, so nothing about it ever
crosses onto the Qt/GUI thread. `read()`/`get_info()` are just
lock-protected attribute reads the GUI thread polls exactly like it
already polls `CameraStream`.

**Verified** (offscreen, against a local test page -- a `<video>`
fed by `canvas.captureStream()` from a `requestAnimationFrame` animation,
not a real VDO.Ninja stream, same "synthetic data instead of real
hardware" approach used for ChArUco/scale testing earlier in this
project): status progresses connecting -> waiting for video -> live;
native resolution exactly matches the synthetic canvas size; `read()`
returns real, changing frames and an independent copy each call (never
a shared reference); `release()` cleanly stops the thread; the object is
reusable after `release()`; and a bad/unreachable URL is reported as an
error status, never a hang or an unhandled exception. A real VDO.Ninja
viewer URL still needs a manual check on your end -- that requires an
actual publishing device.

**Bug found on the first real-stream attempt, now fixed**: a real
VDO.Ninja connection would appear briefly on the streaming end, then
immediately disconnect. Root cause: `LiveCameraTab._update_frame()`'s
consecutive-failure auto-disconnect (originally written for USB, where
a `None` read always means something's actually wrong) was being
applied to VDO.Ninja too -- at `VDO_POLL_INTERVAL_MS=80` and the
existing `max_consecutive_failures=60`, that's only **4.8 seconds**
before giving up, nowhere near enough time for a real WebRTC handshake
(page load, signaling, ICE negotiation) to finish. `VdoNinjaSource`
returning `None` while legitimately still in its normal "connecting"/
"waiting for video" states was being torn down mid-handshake. Fixed:
for VDO.Ninja, `_update_frame()` no longer disconnects on a bare `None`
read at all -- it only auto-disconnects on `VdoNinjaSource`'s own
explicit `"error: ..."` status, which already has its own generous
internal timeouts (30s page load / 30s for a `<video>` element / 60s for
live video) before it ever reports one. Also reinstated the console/
page-error/failed-request diagnostic hooks `../test_code.py` already had
(printed, since this now runs headless inside a GUI rather than a
terminal CLI loop) -- omitted when this was first adapted, which had
left no visibility into *why* a connection failed beyond a generic
status string. Verified directly with a synthetic page whose `<video>`
doesn't go live until 6 seconds after connecting (deliberately longer
than the old 4.8s window): the connection now survives that delay
without being auto-disconnected, and correctly reaches "live" once real
frames start arriving.

**Second bug found on the same real-stream attempt, also fixed**: once
connected, the captured video was locked to a far lower resolution than
the real stream. Root cause, part one: `_JS_BIND_CANVAS` sized the
capture `<canvas>` **once**, at the moment the `<video>` element first
had *any* nonzero frame -- but WebRTC senders commonly ramp up from a
small placeholder/negotiation frame to the real negotiated resolution
over the following seconds, and `videoWidth`/`videoHeight` change to
reflect that. The canvas never got resized afterward, so every
subsequent frame was silently captured (and upscaled-by-drawImage) at
that first, low, size. Fixed: `_JS_CAPTURE_FRAME` now compares the
canvas's current size against `video.videoWidth`/`videoHeight` on
**every** capture and resizes before drawing if they've diverged, and
returns its actual captured dimensions each time so `get_info()`'s
reported native resolution stays live-accurate instead of frozen at the
bind-time value.

Root cause, part two -- found and confirmed against a real, live stream
(the user supplied an actual `https://vdo.ninja/?view=...` link, not a
synthetic page, for this specific investigation): even with part one
fixed, resolution still plateaued around 450x800 against a real
portrait 1080x1920 source. VDO.Ninja (like many WebRTC/simulcast
platforms) selects which quality layer to *send* a given viewer based on
how large that viewer's `<video>` element is actually **displayed** --
confirmed directly by forcing the video element to a large CSS box and
watching the negotiated resolution climb in response, converging on
whatever the box's constraining dimension was. A small/mismatched-aspect
box (the original 1280x800 landscape viewport, letterboxing a portrait
stream) was an implicit "I only need low quality" signal. Fixed:
`_JS_MAXIMIZE_VIDEO_ELEMENTS` forces every `<video>` element to a large
(1920x1920) CSS box immediately after it's found -- before even waiting
for live video, so the size hint is in place *during* negotiation, not
applied only after the fact -- and the browser context's own viewport
was enlarged to match (1920x1920, chosen square specifically so neither
landscape nor portrait streams get constrained by a mismatched aspect
ratio). Verified against the real stream: resolution now ramps all the
way to the source's true 1080x1920, and the captured frames are real,
correctly-oriented, non-black images (visually confirmed).

**FPS, also measured against the real stream**: at full 1080x1920,
PNG-encoding every frame capped achieved fps around 12-13 even with
`requested_fps=30` -- confirmed PNG encode time was the bottleneck, not
the polling rate. Switched to JPEG (quality 0.85): the source is WebRTC
video, already lossy-compressed before it ever reaches this canvas, so
this doesn't introduce a first generation of loss, only a second one on
top of a signal that was never lossless to begin with -- reasonable once
confirmed with the user that lossless wasn't required here (unlike the
USB path, which stays lossless deliberately). This raised achieved fps
to ~19-20. Tested whether a *lower* JPEG quality (0.6) would push fps
higher still: no measurable difference (~19-20fps either way) -- meaning
the real ceiling at this resolution is the fixed per-frame
`page.evaluate()`/CDP round-trip cost itself, not encoding time or
payload size, so 0.85 was kept (no fps benefit to trading away more
fidelity than that). ~19-20fps is short of a genuine 60fps at full
1080p with this Playwright-eval-based capture technique -- an honest
ceiling of the approach, not a tuning problem; the "Processing FPS"
spinbox on Live Camera now allows requesting up to 60 (it's a ceiling on
the capture loop, never a guarantee -- requesting more than what's
achievable is harmless, the loop just runs as fast as it actually can).

**Third and fourth bugs, found when the user still saw ~10fps after all
of the above**: two more mistakes, both now fixed.
1. The GUI's own display-refresh timer (`LiveCameraTab.timer`) was
   started at a **fixed** 80ms (12.5Hz) for VDO.Ninja, completely
   decoupled from the "Processing FPS" setting -- so even once the
   capture loop itself reached ~19-20fps, the GUI only ever polled
   `read()` 12.5 times/sec. Since `read()` returns only the *latest*
   frame (not a queue), polling slower than the real capture rate means
   real, distinct frames get silently skipped before the GUI ever sees
   them, not just delayed -- so the displayed/relayed rate was capped
   at ~12.5fps regardless of how fast frames were actually being
   captured underneath. Fixed: `connect_vdo_ninja()` now computes the
   poll interval directly from the configured Processing FPS
   (`max(10, 1000/fps)` ms), so raising that spinbox actually raises
   what you see, up to the real ~19-20fps capture ceiling. Verified with
   a unit test asserting the timer's actual interval matches what a
   given fps setting implies (e.g. 50fps -> 20ms, 10fps -> 100ms), no
   network required.
2. "Show Undistorted" and the crop sliders were both nested inside
   `usb_group`, hiding them entirely whenever VDO.Ninja was the active
   source -- even though both are fully source-agnostic (each profile,
   USB or VDO.Ninja, has its own lens calibration and crop; the
   underlying logic already resolved the active profile correctly, only
   the *widgets* were hidden). Moved both into the shared controls area
   between the two source-specific groups. Verified directly: neither
   widget is a descendant of `usb_group` anymore, and both are visible
   in both source modes.

**On capturing the stream more directly instead of through a headless
browser**: asked directly by the user, worth answering plainly. What
this module does -- render the page in a real (headless) browser,
screenshot the video element via canvas, ship that over Chrome's
DevTools Protocol -- is not the only possible approach. VDO.Ninja is
WebRTC; a native Python WebRTC client (e.g. `aiortc`) could in principle
receive and decode the actual media track directly, with no browser
rendering or per-frame CDP round-trip in the loop at all, which would
remove the ~19-20fps ceiling measured above entirely. That was not
attempted here: it would mean re-implementing VDO.Ninja's own
signaling handshake (SDP offer/answer, ICE, whatever its specific
websocket protocol looks like) from scratch in Python, without a
reference implementation to adapt the way this module adapted
`../test_code.py`'s technique -- a substantial new piece of work with
real uncertainty about whether it would even succeed, not a quick swap.
Worth pursuing if ~19-20fps at full resolution genuinely isn't enough
for the actual measurement need, but that's a scope decision, not
something to start without discussing it first.

**Calibration profiles** (`calibration_profiles.py`): a profile is
`{id, source_type, name, alias, camera_role, device_path, device_name,
fourcc, calibration_model, detector_type, profile_version, resolution,
orientation, crop_percentages}`, tracked in `calibration_profiles.json`,
with a separate "active profile" remembered per **key** (see "Two USB
cameras" below for why the key is more specific than `source_type` now)
-- switching which USB profile is active never touches which VDO.Ninja
profile is active, and vice versa. The actual calibration *data* for a
profile is never embedded in that
registry file -- each profile gets its own `profiles/<id>/` folder, and
`distortion.py`/`scale.py`/`circle_config.py`/`background_reference.py`
are handed that folder's paths via the `path=` parameter each of those
modules *already* accepted (added back in their original stages for
exactly this kind of override) -- **none of their internals changed at
all** to support profiles. The Calibration tab gained a "Profile" picker
+ "New Profile..." at the top; every capture/save/load button below it
already routes through the active profile's paths. Creating a VDO.Ninja
profile is an explicit action (not auto-created on first connect) --
name it, and it becomes active for that source immediately.

Usable ROI (the crop sliders on Live Camera) moved from a flat global
setting into each profile's own record too, for the same isolation
reason -- `LiveCameraTab.reload_crop_from_profile()` reloads them
whenever the active source or profile changes (on the source combo, and
via `main_window.on_active_profile_changed()` when Calibration's picker
changes it), and `_save_crop_settings()` persists back into whichever
profile is currently active, never a shared file both sources read.

**Migration, not a fresh start**: the very first time this profile
system runs (`calibration_profiles.ensure_default_usb_profile()`, called
once from `MainWindow.__init__`), if the original flat
`calibration_data.json`/`scale_calibration.json`/`circle_config.json`/
`background_reference.npy`/crop settings exist from before profiles
existed, they're copied (never moved -- the originals stay in place,
now simply unread) into a new profile named "USB Camera", made active
for `"usb"`. This actually ran for real against this project's own
already-captured 15-view calibration while building this feature (not
just in a sandboxed test) -- confirmed the migrated profile's
calibration/scale/background files are byte-identical to the originals,
the profile is correctly active for `"usb"` with `"vdo_ninja"` still
unset, and the original flat files remain untouched on disk afterward.

**Never applying one source's calibration to the other's frames**: this
isn't just a warning-based check -- it's structural. Every calibration
load in Calibration/Detection resolves the active profile via
`main_window.active_profile_key` first, then loads *that* profile's
files; there is no code path that reads a USB profile's calibration
while VDO.Ninja is the active source, or vice versa -- and, since
`active_profile_key` (unlike `active_source_type`) resolves per
*physical* USB camera, the same guarantee extends to the two USB cameras
never reading each other's calibration either (see "Two USB cameras"
below). What's still a runtime check (same pattern used since Stage 3): a
live frame's resolution not matching its profile's saved calibration
resolution still disables and explains, rather than silently sampling
wrong (`cv2.remap` does not error on a size mismatch). `orientation` is
tracked per profile per the plan but not yet actively enforced -- no
rotation control exists in the UI yet, so it's currently always
`"normal"`; noted here rather than overclaiming a check that isn't real
yet.

### Two USB cameras: stable identity, roles, and per-camera profiles

The app supports two distinct physical USB cameras at once (though only
one is ever connected/streaming at a time, same "never two streams"
guarantee as USB vs. VDO.Ninja above): the original monochrome
U20CAM-9281M, and an ELP-USBFHD01M-BL170 (1080p color, MJPEG, 170-degree
fisheye lens). Both show up as separate entries in Live Camera's
"Camera:" dropdown.

**Why not just use OpenCV's index to tell them apart**: Windows can and
does reassign which index maps to which physical device across reboots
or replugging -- index 0 today might be a different camera than index 0
tomorrow. `camera_identity.py` reads each device's DirectShow
`DevicePath` (the same property-bag mechanism `pygrabber` already uses
internally for friendly names, just a different key) -- stable across
reboots/replugging into the same port for most UVC devices. The rare
driver that doesn't expose one falls back to matching by friendly name
instead (documented limitation: two identically-named unnamed devices
can't be told apart this way). Either way, the index itself is only ever
used as a momentary connect parameter, resolved fresh on every probe --
never persisted as identity.

**First connect to an unrecognized camera prompts immediately** for an
alias (free text, defaults to the DirectShow friendly name) and a role
(constrained to exactly two choices: "Monochrome FTIR Camera" or "Color
FTIR Camera") -- confirmed with the user this happens automatically, not
via a separate manual "New Profile" step, and happens *before*
`stream.open()` is called so the very first connection already uses a
sensible parameter set. The role picks a *default* preselection in the
new "Format:" dropdown described below and sets `calibration_model`/
`detector_type`:

| Role | Default format (preselected, not fixed) | `calibration_model` | `detector_type` |
|---|---|---|---|
| `monochrome_ftir` (U20CAM) | 1280x800 @ 120fps, MJPG | `pinhole` | `grayscale` |
| `color_ftir` (ELP) | 1280x720 @ 60fps, MJPG | `pinhole` | `color` |

**Resolution/pixel-format/fps are fully adjustable, not hardcoded** -- a
"Format:" dropdown next to the "Camera:" selector is populated from
`camera.list_device_formats()`, the same real DirectShow enumeration
AMCap's own format dialog uses (gathered in the same background probe
pass as the device list, since building the enumeration is noticeably
slower than a plain friendly-name/identity lookup). It lists every
resolution/pixel-format/fps combination the connected driver actually
advertises -- e.g. the real ELP used during development advertises 14
distinct combinations (MJPG *and* YUY2, from 320x240 up to 1920x1080) --
and whatever's selected there, not a fixed per-role lookup, is what
`connect_camera()` actually requests. The role table above only affects
which entry gets *preselected* for a never-before-seen camera; picking a
different entry and connecting works immediately, and that choice is
remembered (`calibration_profiles.update_profile_connect_format()`) so
reconnecting later reselects the same format instead of reverting to the
role default.

**The ELP's default was changed from 1920x1080@30fps to 1280x720@60fps**
after a real "black video" report. Root cause, confirmed directly against
the real ELP hardware: it was not actually a decode/corruption bug --
`camera_controls.get_all()`/cranking Brightness on a *genuinely connected*
camera fixed it immediately (real sensor noise was there all along, just
very underexposed at the driver's default Brightness -- already fully
adjustable via the existing camera-controls panel, see "Camera settings
(Stage 2)" below; Brightness/Contrast/Saturation/Sharpness/Gamma/White
Balance are all already exposed there, generically, for whichever camera
is currently connected, no per-role changes needed). The deeper issue turned out to be a **wrong physical
device**: the "Fisheye 170" alias had been assigned, via the naming
dialog, to the laptop's built-in webcam rather than the external ELP --
confirmed by unplugging the ELP and watching which DirectShow device
actually disappeared. The built-in webcam produced a perfectly flat,
brightness-unresponsive frame (`std=0.00` across many frames, exact same
pixel value every time) -- consistent with Windows Camera privacy
permissions blocking a desktop app from that specific device, not
anything this app's capture code controls. Once reassigned to the
correct `device_path`, the real ELP produces genuine live video (varying
per-frame noise, `std` in the tens) at both 1280x720@60fps and
1920x1080@30fps -- 720p60 is simply the steadier of the two and is now
the default; either remains one dropdown selection away.

**Both roles use the pinhole calibration model, not fisheye** -- a
deliberate simplification, decided directly with the user after finding
that `cv2.fisheye.calibrate()`'s 4-parameter model was numerically
unstable at realistic (~20-25) ChArUco view counts (higher-order
coefficients k3/k4 would occasionally blow up to physically implausible
values even though fx/fy converged fine -- confirmed this wasn't fixable
by just capturing more views, and traced to the higher-order terms being
poorly constrained without views that reach deep into the lens's extreme
periphery). Since the ELP's actual working area for this app isn't
distorted enough to need full 170-degree correction, both cameras now
share the exact same `distortion.py` pinhole/ChArUco path. A separate,
tested fisheye calibrator (`fisheye_calibration.py`) existed for
if/when edge-accurate correction was ever needed, but no tab ever
instantiated it, and it was later removed (see "Files" above) -- revive
from git history if that need ever actually comes up.

**Reconnecting an already-known camera never re-prompts** -- its
existing profile is resolved by `device_path` (self-healing: a profile
originally matched by name, e.g. from before `device_path` was known,
gets it filled in the first time it's actually seen) and reused. Its
saved calibration, scale, ROI, and detector settings are exactly as left
last time.

**Switching cameras while one is connected disconnects it first** -- the
same guarantee `_on_source_type_changed` already gives USB vs.
VDO.Ninja, extended one level down: selecting a different entry in the
"Camera:" dropdown while a USB camera is streaming calls
`disconnect_camera()` before the new one can be connected, so there is
never a moment where two `CameraStream`s are open at once.

**Active-profile keys, generalized**: before this stage, `active_by_source`
had exactly two fixed keys, `"usb"` and `"vdo_ninja"`. With two physical
USB cameras, `"usb"` alone isn't specific enough -- `calibration_profiles.
profile_key()` now computes `f"usb:{device_path}"` (or
`f"usb:name:{device_name}"` for the rare no-DevicePath fallback) for USB
profiles, `"vdo_ninja"` unchanged otherwise. `main_window.
active_source_type` still exists and still means exactly "usb" or
"vdo_ninja" (it only drives which UI control group is shown); the new
`main_window.active_profile_key` is what every profile lookup actually
uses, and is the one that diverges between the two USB cameras. An old
registry saved under the pre-existing flat `"usb"` key is migrated to the
new key shape in memory on load (never a destructive rewrite), so the
already-migrated real U20CAM profile keeps working without any manual
step.

**Verified end-to-end** (offscreen Qt, two fake camera identities
standing in for real ELP/U20CAM hardware -- neither was available to
test against directly): both show as "not yet set up" in the dropdown on
first probe; connecting each triggers the naming dialog exactly once and
never again on reconnect; the two get distinct `device_path`-based
profile keys and never share a profile; connect parameters match the
role table above exactly; selecting a different camera while one is
connected disconnects it first; the Detection tab correctly routes the
ELP (`source_type="usb"`, `detector_type="color"`) to the color
detector/UI rather than the grayscale one; and the real, already-migrated
U20CAM profile from before this stage is untouched by any of the above
(verified via a full backup/restore of the actual project's
`calibration_profiles.json` around the test, confirmed byte-identical
afterward).

## Calibration tab (Stage 3)

Lens distortion correction via a ChArUco board (a checkerboard with
embedded ArUco markers) -- chosen over a plain checkerboard because
partial/angled views still contribute usable corners, since each square
is individually identifiable by its marker; a plain checkerboard needs
the whole grid unambiguously visible in every captured frame.

**Board parameters are input fields**, not hardcoded, in `distortion.py`'s
`CharucoCalibrator` and the tab's UI: squares (X/Y), square length (mm),
marker length (mm), and the ArUco dictionary. The board currently in use
is 8x8 squares / 20mm squares / 15mm markers / `DICT_5X5_100`, but
switching to a 6x6 board or a different dictionary later just means
different input values -- no code changes.

Workflow:

1. Set the board parameters to match your physical board and click
   "Create / Update Board" (this also clears any previously captured
   views, since they were matched against the old board's geometry and
   aren't valid for a different one).
2. Point the camera at the board. The preview updates live (reading
   `main_window.latest_frame`, not a second competing camera read) and
   overlays detected markers/corners plus a running corner count, so you
   can see whether the board is being recognized *before* capturing a
   view.
3. Click "Capture Frame" at a range of different angles/distances/
   positions -- **at least `distortion.MIN_CALIBRATION_VIEWS` (15)**,
   deliberately varied: close and far, tilted in different directions,
   covering the frame's edges and corners, not just near-identical
   flat-on shots of roughly the same spot. Each capture is verified to
   have enough corners before being added; a rejected capture tells you
   so rather than silently accepting a bad view. The on-screen help text
   under the capture buttons repeats this.
4. Once at least `MIN_CALIBRATION_VIEWS` views are captured, "Run
   Calibration" becomes available. It reports the reprojection error
   (lower is better; well under 1.0 pixels is a good sign) and saves
   camera matrix + distortion coefficients to `calibration_data.json`
   -- unless the result fails the sanity check below, in which case
   nothing is saved and the previous (still valid) calibration, if any,
   stays in place.

**Why 15, not the original 4, and a sanity check on top**: too few
views -- or enough views that are too similar to each other -- badly
underconstrains the model `cv2.calibrateCamera` fits (fx, fy, cx, cy
plus 5 distortion coefficients, 9 free parameters). A real run of this
tab with only 4 similar-angle captures produced `fx≈5517`/`fy≈5572`
(implies a telephoto FOV on what's actually a normal-FOV lens) and
`k3≈-39896` (real lenses land within a couple orders of magnitude of 1)
-- a reprojection error that looked fine (0.28px) on those exact 4
views, while producing wild, non-physical remapping everywhere else:
horizontal lines scaling oddly, apparent rotation, shrinking -- i.e.
undistort made the image *worse* than doing nothing. Reprojection error
alone doesn't catch this kind of overfitting. Two changes address it:
  - `CharucoCalibrator.calibrate()` now seeds `cv2.calibrateCamera` with
    a sane initial guess (`fx=fy=image width`, centered principal point
    -- the standard rule-of-thumb starting point for a non-fisheye lens)
    and passes `CALIB_FIX_ASPECT_RATIO` (this sensor has square pixels,
    so fx and fy should stay equal throughout the solve) instead of
    starting from OpenCV's own default guess, which is what let fx/fy
    diverge in the first place.
  - `distortion._sanity_check_calibration()` checks the *result* against
    generous but physically-grounded bounds (focal length vs. image
    size; distortion coefficient magnitudes) regardless of how it was
    produced, and blocks the save with an on-screen explanation if it
    looks overfit rather than silently trusting a low reprojection error.
    Verified directly: correctly flags the real bad 4-view result above,
    stays silent on a synthetic well-conditioned 18-view result whose
    recovered camera matrix landed within ~1% of the known-true values
    used to generate it, and (via the tab itself, offscreen) both leaves
    `calibration_data.json` untouched when a forced bad calibration is
    attempted and successfully saves a good one.

**Maximizing retained pixels**: `build_undistort_maps()` already used
`alpha=1` in `getOptimalNewCameraMatrix` (keeps the full field of view,
accepting black corners, rather than `alpha=0` which crops in to
guarantee no invalid pixels) -- that choice was already correct and
unrelated to the overfitting bug above; a bad distortion model just made
the maps it built garbage regardless of alpha. The crop sliders below
exist to trim the specific low-coverage border pixels left over from
that `alpha=1` choice, independently of this fix.

**Toggle it on to see it working**: Live Camera has a "Show Undistorted"
checkbox. It's preview-only -- applies `distortion.undistort_with_maps()`
to the *displayed* frame after loading the saved calibration, and never
touches the raw frame other tabs see (`main_window.latest_frame`, or
whatever a future Detection stage reads). If the live resolution doesn't
match the resolution the calibration was computed at, it's automatically
disabled with a clear message rather than silently showing a
wrong/cropped result -- confirmed directly that `cv2.remap` does **not**
error on a size mismatch, it just samples into whatever frame it's given
using the maps' own dimensions, so this had to be checked explicitly
rather than relying on an exception.

**Crop after undistort**: right below the toggle, four independent
sliders (Top/Bottom/Left/Right, each 0-45% of that dimension) trim edge
pixels off the *displayed* undistorted frame -- the region built up by
`alpha=1` in `build_undistort_maps()` (kept to preserve full field of
view) includes the pixels the calibration had the least data coverage
for, where any remaining warping is most visible. Independent per side
rather than one shared margin, since distortion isn't necessarily
symmetric (e.g. a slightly off-axis lens/sensor). Percentage-based, not
fixed pixel counts, so it holds up across different resolutions.
`distortion.crop_edges()` clamps each axis (top+bottom, left+right) so at
least 10% of that dimension always remains, protecting against a
degenerate empty crop if both sliders on one axis are pushed high at once
-- confirmed directly, including the clamped case. Values persist to
`app_config.json` (loaded on startup, saved when you release a slider,
not on every drag tick).

**Verified end-to-end** (offscreen Qt mode, since no physical board is
available in this dev environment): generated a real ChArUco board image
with OpenCV itself, perspective-warped it a few different ways to
simulate multiple captured views, ran it through the actual tab code
(`create_board` → live detection → `capture_frame` × `MIN_CALIBRATION_VIEWS`
→ `run_calibration`) and got a real reprojection error and a saved
`calibration_data.json`;
connected the dev environment's camera at the same resolution as that
calibration, confirmed "Show Undistorted" stayed enabled, adjusted all
four crop sliders and confirmed the displayed frame's shape actually
changed and the saved config round-tripped correctly. Separately,
confirmed the resolution-mismatch path (calibration done at a different
size than the live feed) still auto-disables rather than silently
showing a wrong frame.

Two real implementation snags worth knowing about:

- `aruco.drawDetectedCornersCharuco` raised a `cv2.error` shape-assertion
  under partial/ambiguous detection even when the corners/ids arrays had
  matching Python `len()` -- the mismatch was at the underlying C++
  `Mat.total()` level, not visible from Python. Since that call is only a
  visual annotation (the actual calibration math uses the raw corner/id
  arrays directly via `matchImagePoints`, not the drawing call), it's
  wrapped so a drawing failure is caught and skipped rather than treated
  as detection having failed.
- **If markers detect but the corner count stays at 0**: OpenCV changed
  the ChArUco corner/marker layout convention around version 4.6. A board
  generated by another tool or an older OpenCV version may still use the
  old ("legacy") layout -- markers still detect fine (dictionary lookup,
  layout-independent), but corner interpolation silently returns zero
  against the wrong expected geometry. Reproduced this exact symptom
  directly (32 markers, 0 corners) before fixing it. The Calibration
  tab's "Legacy pattern" checkbox (next to the dictionary dropdown) toggles
  `CharucoBoard.setLegacyPattern()` -- if you hit this, check it and click
  "Create / Update Board" again. The live status line also detects this
  specific combination (markers > 0, corners == 0) and tells you to try
  it, rather than just showing a bare "0 corners" with no explanation.

## Physical scale calibration (Stage 4)

A "Physical scale" section below the ChArUco board section on the
Calibration tab. Converts pixel distances into real-world mm: click two
points a known distance apart (a ruler, or two marks of known
separation), enter that distance, get `mm_per_pixel`. This is a simple
linear scale, not a full homography/perspective rectification -- accurate
as long as whatever you measure later is roughly in the same plane, at
the same distance from the camera, as when you did this calibration
(true for the actual FTIR contact-surface setup this is built for).

**Requires Stage 3 first, enforced, not just documented**: the section
starts disabled with "Complete lens-distortion calibration above first."
until `distortion.load_calibration()` finds a saved calibration --
distances are only physically meaningful on the *undistorted* image,
since lens distortion changes apparent distances non-uniformly across
the frame. Same explicit resolution-match check as the "Show Undistorted"
toggle (not an exception -- confirmed back in Stage 3 that `cv2.remap`
doesn't raise one for a size mismatch, it just samples wrong).

Workflow:

1. "Capture Frame for Scale" -- freezes the current frame after applying
   `distortion.undistort_with_maps()` **and** the same edge crop
   currently set on Live Camera's "Crop after undistort" sliders
   (`main_window.crop_percentages`, kept in sync by those sliders and
   read here via `distortion.crop_edges()`) -- the scale reference
   matches what you're actually looking at, and avoids the same
   least-trustworthy border pixels the crop sliders exist to trim in the
   first place. Frozen deliberately: clicking precise points on a
   constantly-updating live view would be imprecise.
2. Click two points on the frozen image (each drawn as a red marker, with
   a connecting line once both are placed). A third click starts a new
   pair rather than adding a third point. A click that lands outside the
   image (in the letterboxed margin around it, if the image's aspect
   ratio doesn't match the label's) is rejected with a message rather
   than silently recorded as a bogus point.
3. Enter the known real-world distance between those two points in mm.
4. "Compute & Save Scale" -- saves `mm_per_pixel`, the pixel distance
   measured, and the image size to `scale_calibration.json`.

"Retake" discards the frozen frame and any placed points, re-enabling
"Capture Frame for Scale" for another attempt.

**Verify Scale**: once a scale is saved, "Verify Scale (click 2 points)"
becomes enabled. Click it, then click two *new* points on the same frame
-- of a distance you separately know, e.g. two other marks on the same
ruler -- and it reports the measured real-world distance using the
*already-saved* scale (`scale.measure_distance_mm()`), so you can sanity
check it before trusting it. Verify-mode markers are drawn in orange
instead of red, as a visual reminder you're in a different mode; nothing
clicked in verify mode ever touches the saved calibration file --
confirmed directly (byte-for-byte identical file contents before and
after a round of verify clicks). "Retake" exits verify mode back to
calibrate mode along with everything else it resets.

**Coordinate mapping, done carefully**: the frozen frame is scaled to fit
the preview label (`Qt.KeepAspectRatio`-equivalent, computed explicitly
here rather than via `QPixmap.scaled()`'s own internal choice) and
centered if its aspect ratio doesn't exactly match the label's. Click
coordinates arrive in label-widget pixels and have to be mapped back
through that same scale and centering offset to get original-image
pixels -- done explicitly with the *same* scale/offset values used to
render the frame, rather than recomputing them separately at click time,
so the two can never drift out of sync with each other.

**Verified end-to-end** (offscreen Qt mode): confirmed the section stays
disabled with no distortion calibration present, and enables once one
exists; built a real ChArUco calibration the same way as the Stage 3
tests, then ran the actual tab code (`capture_scale_frame` ->
`_on_scale_image_clicked` x2 -> `compute_and_save_scale`) with two points
clicked at known image coordinates -- the coordinate round-trip was
*exact* (200.0, 300.0 and 700.0, 300.0 recovered precisely from their
widget-space equivalents, not just approximately), and 500px measured as
100mm produced exactly 0.20000 mm/pixel. Also verified: a third click
correctly discards the first two and starts a new pair; an out-of-bounds
click is rejected without being recorded; "Retake" fully resets state and
re-enables capture; and the resolution-mismatch gate correctly refuses to
capture when the live frame's resolution doesn't match the saved
distortion calibration's, without touching `scale_frame` at all.

## Detection tab (Stages 5-6 live rebuild, plus part of 8)

**Now switches its entire detector, not just its data source**, based on
the *active profile's* `detector_type` -- **not** `main_window.
active_source_type` directly, since a plain "usb" no longer implies
grayscale now that the color ELP camera exists alongside the monochrome
U20CAM (both `source_type="usb"`, different `detector_type`). Profiles
with `detector_type="grayscale"` (the U20CAM) get the background-
subtraction pipeline described below unchanged; profiles with
`detector_type="color"` (VDO.Ninja, and the ELP) get an HSV color
detector (`color_detection.py`) on a second page of a `QStackedWidget`,
with its own color preset/hue/weak/core/threshold controls and its own
"Processing rate" throttle (color detection is heavier, and unlike
grayscale isn't tied to a background reference at all -- the whole
"Background reference" section hides whenever `detector_type` isn't
`"grayscale"`, since that detector doesn't do background subtraction).
Both detectors
produce `detection.Fragment` lists wrapped into the same
`regions.DetectedRegion` via `regions.from_fragment()` before rendering,
so the report/panel code below is written once, not duplicated per
detector -- see "Camera sources and calibration profiles" above for how
`color_detection.py` reuses `detection.detect_fragments()` verbatim
(same no-blur/no-closing/no-circularity-assumption guarantees, just fed
a color-confidence array instead of a brightness-difference one) and how
its hysteresis (strict core + broader pale weak region) works.

**Rebuilt from an earlier single-shot version.** That version ran once
per button click and drew each candidate group's *expected*-size ROI as
a circle -- reasonable as an internal grouping aid, but shown live it
looked like "detected circular light" even where the real illumination
was smaller, absent, or a different shape entirely, since real contact
light is not necessarily circular. This version fixes both problems:
continuous live updates, and detected regions shown only as their actual
traced shape, never a circle. `assignment.py`/`measurement.py` (the
fragment-to-circle grouping + area-coverage code from the previous
version) were kept, untouched and still correct, for a time in case
grouping got re-enabled on top of shape-aware detection -- ultimately
removed once per-pad measurement was built a different way instead (see
"Recording workflow" further down).

**Continuous, without a second camera connection**: `DetectionTab` owns
a `QTimer` (400ms -- slower than other tabs' timers since each tick now
does a background diff + connected-components + six panel renders) that
reads `main_window.latest_frame` -- the exact same shared reference
`LiveCameraTab` already writes and `CalibrationTab` already reads, never
a new `CameraStream`. A plain repeating `QTimer` already gives "skip
stale frames, process only the newest" for free: Qt doesn't queue up
missed fires for one timer, it just fires again once the event loop is
free, and each tick reads whatever `main_window.latest_frame` currently
holds -- there's never an explicit backlog to build or flush. Every tick
starts with `if not self.isVisible(): return`, so it does no work at all
while another tab is active.

**Background subtraction, not raw brightness thresholds**: a new
"Capture Background Reference" button (`background_reference.py`) saves
the current corrected frame -- undistorted, then cropped to
`main_window.crop_percentages`, the same "usable ROI" restriction used
everywhere else in this app -- verbatim to `background_reference.npy`.
Because it's captured in that same corrected coordinate system, every
live frame lines up with it pixel-for-pixel with no further registration
step. Every tick then computes `compute_difference()`: `current -
background`, clipped at 0 (never negative). Static structure present in
*both* frames -- fixture reflections, LED hot spots, ambient glare baked
into the glass -- subtracts to ~0 and disappears; only genuinely new
illumination (real contact) survives as a positive value. The three
possible/probable/strong thresholds apply to this difference image, not
absolute brightness -- defaults dropped from 40/100/180 to 10/25/50
accordingly, since a real new spot might only be 10-60 counts brighter
than the background, much smaller than typical absolute pixel values.
You capture this with the camera fixed, undistortion always applied
internally, normal LEDs on, exposure/gain locked (Settings tab), and no
intended contact -- the tab's own help text repeats this.

**Fragment definition is unchanged from the previous version** (still
the mechanism that guarantees no dark gap is ever bridged): `detection.py`
runs connected-component labeling exactly once -- now on the difference
image instead of raw brightness, which required zero code changes there,
since it was already written to operate generically on whatever single-
channel array it's given. No blur, no morphological closing/dilation
anywhere. Every fragment is kept, including single-pixel ones. Each
fragment's contour is drawn exactly as `cv2.findContours` traced it --
no circle, no shape assumption anywhere in this module or the tab.

**Grouping was off here, on purpose, for the entire lifetime of this
tab**: nothing is ever drawn or reported as a "circle," only as a
fragment. Every fragment's status in the report is `UNCERTAIN`; there's
no accepted/extra/missing bucketing. The report does include per-
fragment ID, pixel area (mm² too, if a physical scale is saved --
`possible_area_px` gained no new fields here, but `Fragment` gained
`p95_brightness` alongside the existing mean/median/max/min), centroid,
and mean/median/95th-percentile difference intensity, all computed
directly from the difference image, never a blurred copy.

**Six live panels**, each redrawn every tick: the corrected live frame,
the difference image (auto-contrast-stretched *for display only* -- the
actual thresholding always uses the true unstretched difference values,
never the stretched display copy), the three threshold masks, and the
live frame with actual fragment contours drawn over it in one consistent
color (no accepted/extra/unassigned coding, since nothing is classified
yet).

**Double-click any panel to expand it** full-size in place of the grid
(the small panels are otherwise only ~320x220) -- double-click again, or
press Esc, to return. The expanded view keeps updating live from the
same timer tick, it's not a frozen snapshot; whichever panel key is
currently expanded gets mirrored into the large label alongside its
small grid counterpart (the small one is kept current in the background
too, so the grid is never stale when you collapse back to it). This is
an in-window expand, not a separate popout or OS-level fullscreen.

**Verified**: a synthetic background frame with a static bright rectangle
(fixture reflection) and a static ambient glow circle, and a live frame
with the *same* static structures plus one genuinely new, deliberately
non-circular illuminated region (a 10-point star polygon). Confirmed
directly: both static structures cancel to a difference of 0, the star's
true brightness delta (185 counts) survives exactly; `detect_fragments()`
on that difference image finds exactly one fragment whose contour
perimeter is markedly longer than a circle of the same area would have
(469px vs. 241px for an equal-area circle) -- i.e. genuinely not
circle-approximated. Ran the same scenario through the actual tab
(offscreen Qt): confirmed `_tick()` is a no-op while the tab isn't the
visible one; "Capture Background Reference" saves the exact corrected
frame; a background-only frame correctly detects 0 fragments; adding the
star produces exactly 1, matching the standalone module test; switching
back to a background-only frame on the next tick correctly drops back to
0 (proving each tick reflects the newest frame, not a stale one); the
report shows only `UNCERTAIN`, never `ACCEPTED`/`EXTRA`; all six panels
render; and `main_window.latest_frame` is verified byte-identical before
and after processing.

## Recording workflow (done): record now, analyze later, per pad

Everything above assumed live detection was the end state -- watch the
feed, tune thresholds, read the report. In practice that's the wrong
shape for actually collecting FTIR data: you want to record footage
once, while the pad is actually in front of the camera, then go back and
iterate on detection tuning against those exact same saved frames as
many times as needed, without ever needing the physical setup again --
and measure each pad's contact independently rather than getting one
whole-frame report. This section is that: lossless recording, a
recordings library, offline playback + detection, and per-pad ROI
(region of interest) analysis with its own saved, re-browsable results.
Three new tabs (Recordings, Processing, Results); the existing Live
Camera/Detection tabs and everything above are unchanged by any of it.

### Frame dispatcher + threaded capture

Recording needs *every* frame the camera delivers, not just whatever a
15ms GUI polling timer happened to grab -- so capture had to move off
the GUI thread first. `threaded_camera_source.py`'s `ThreadedCameraSource`
wraps an already-open `camera.CameraStream` in a dedicated background
thread that calls `.read()` in a tight loop at the camera's true
delivered rate, duck-typing the same `read()`/`get_info()`/`release()`
shape `LiveCameraTab` already expected (same precedent as
`vdo_ninja_source.VdoNinjaSource` before it) -- zero changes needed to
`_update_frame()` itself. Two separate locks guard it deliberately (not
one): a public `lock` around the real `cv2.VideoCapture` calls (both the
capture loop's reads and `camera_controls_panel.py`'s slider get/set
calls), and a private `_state_lock` around just the last-frame handoff,
so a slider drag never makes the live preview stall waiting on the same
lock as the capture loop's hot path.

`frame_dispatcher.py`'s `FrameDispatcher` (one instance, owned by
`main_window`, created once at startup) is the single point every
captured frame -- USB or VDO.Ninja -- passes through before anything
else sees it. Its `publish()` is called from whichever background
capture thread just produced a frame, never the GUI thread, and does
three things: rejects the frame if it's from a session that's since been
invalidated (defense in depth -- the primary safety mechanism is the
caller already joining the old capture thread before starting a new
one), updates `main_window.latest_frame` for the existing GUI-polling
consumers (Live Camera's own preview, CalibrationTab, DetectionTab --
none of which need a gapless sequence, just "something recent"), and
invokes every subscribed callback synchronously on the capture thread
with the same frame. Subscriber callbacks (the recorder is the first and
only one so far) must stay fast/non-blocking -- typically just a
`queue.put_nowait` into the subscriber's own bounded queue, with the
subscriber counting its own drops, not the dispatcher. A callback that
raises is caught and printed, never allowed to take down the capture
thread.

### Recorder + on-disk recording storage

`recorder.py`'s `Recorder` subscribes to the dispatcher for the exact
duration of `start()`..`stop()` and drives a real `cv2.VideoWriter` --
frames published while a camera is merely connected but not recording
are never seen by this module at all, so its own frame-acquisition
counts are honest by construction, not an after-the-fact filtered count.

**Codec is chosen automatically from the active profile's `camera_role`,
never asked for directly** -- this is the direct, evidence-based result
of real-hardware validation against real footage from both cameras, not
a synthetic benchmark:

| `camera_role` | Codec | Why |
|---|---|---|
| `monochrome_ftir` (U20CAM) | **FFV1** | HFYU corrupts grayscale content by ±1 per pixel (confirmed via round-trip test); FFV1 doesn't. |
| `color_ftir` (ELP) | **HFYU** | FFV1's color encoder can't keep up -- 67-71% frame-drop rate measured at both 1080p30 and 720p60; HFYU keeps up with zero drops and round-trips real ELP footage byte-identical. |

Both are genuinely lossless -- this isn't a quality tradeoff, each
camera role simply breaks a *different* one of the two obvious lossless
codec choices in practice, discovered by actually recording real footage
and diffing it back, not by reading a compatibility table. A separate
`quality_mode="lossy_prototype"` override exists for either role
(MJPG), explicitly tagged in `metadata.json` and never treated as
measurement-grade by anything downstream -- for quick framing/setup
checks where disk space matters more than fidelity.

`recording_store.py` owns the on-disk schema:

```
recordings/<recording_id>/
    recording.<ext>            -- written by recorder.py
    metadata.json               -- camera/calibration/recording blocks
    frame_index.jsonl           -- one row per frame (timestamp, write outcome)
    calibration_snapshot/
        calibration_data.json
        scale_calibration.json
        circle_config.json
        background_reference.npy   -- only if present at record time
    .incomplete                 -- present only while open; removed only on a verified-clean finalize
```

**Self-containment, structurally enforced, not just documented**:
`calibration_snapshot/` is populated once, at record start, by copying
the profile's *actual* files as they exist right now
(`snapshot_calibration()`) -- never re-resolved from the live profile
again afterward. Every offline read (Processing tab's playback,
detection, ROI analysis) loads from a recording's own snapshot only,
never the live profile -- so editing or even deleting the source profile
later can never change what an old recording measures against. The same
`.incomplete`-marker crash-safety pattern Stage-3-era calibration
sanity-checking established is reused here: removed only after
`finalize_recording()` succeeds, so a process killed mid-recording
leaves unambiguous evidence, and reused again below for analysis runs.
`DiskSpaceEstimator`/`has_minimum_recording_headroom()` refuse to
start (or keep extending) a recording once free space drops below a
safety margin, rather than finding out from a failed write partway
through a long session.

### Recordings tab

Browse everything on disk under `recordings/`: rename, notes, protect
(blocks delete until explicitly unprotected), delete, open the folder in
the OS file browser, and **Import Video** for footage that didn't come
from this app's own recorder (reads the file's real resolution/fps/frame
count directly rather than trusting a filename, warns if the selected
camera profile's saved calibration resolution doesn't match, and always
tags imported footage `quality_mode="lossy_prototype"` -- this app has
no way to verify an externally-sourced file's lossless fidelity, so it's
never silently treated as measurement-grade). **Open in Processing**
switches to the Processing tab with that recording loaded.

### Processing tab: offline playback, full-frame detection, per-pad ROI analysis

Playback of a saved recording with zoom (up to 400%, drag/scrollbars to
pan), speed control, and double-click either video panel to expand it
full-size (hides the sibling panel *and* the detection/ROI controls
below it, so the expanded panel actually gets the freed space rather
than just the sibling's half of the row).

**Full-frame detection reuses the exact same code DetectionTab's live
loop calls** -- `detection_pipeline.run_detection()` was extracted
specifically so both call sites share one implementation, resolving
`detector_type`/config/background from a recording's own calibration
snapshot instead of the live profile (see self-containment above), with
zero duplicated detector logic.

**Per-pad ROI (region of interest) analysis** is built on top of that,
not a second detector: draw a rectangle around each expected pad (drawn
deliberately *larger* than the pad's real contact area -- it's a search
boundary, never counted as contact itself), and detection then runs
**independently within each ROI's own crop** -- crop the corrected frame
to the rectangle first, then call the identical `run_detection()` on
just that crop, rather than detecting once on the whole frame and
intersecting the result with each ROI afterward. This distinction is
the whole point, not a style preference: the color detector's hysteresis
keeps a whole connected component alive if *any* pixel in it reaches the
"strong" confidence tier -- so a crop-after-the-fact approach could let
a weak-only blob inside one pad's ROI survive only because a strong
pixel *outside that ROI, elsewhere in the frame* happened to share its
connected component. Cropping first makes that structurally impossible:
the crop simply never contains those outside pixels for
connected-components to see. Verified directly with a constructed
reproduction proving the leak is closed.

Coverage percentage is always computed as `detected_contact_area_mm2 /
expected_area_mm2`, where `expected_area_mm2` is a number you enter per
ROI (the pad's real physical contact area) -- **never** against the
ROI rectangle's own area, which is deliberately oversized as a search
margin; using it as the denominator would silently understate coverage
by however much margin was drawn. Coverage is `None` (shown as N/A),
never a fabricated `0`, whenever either the physical scale or the
expected area isn't configured.

Other ROI-analysis details worth knowing:
- **Overlapping ROIs are allowed but block Range/Full analysis** (a
  pixel could otherwise get double-counted toward two pads) -- shown as
  a prominent warning; current-frame preview still runs to help you fix
  it, but its numbers are explicitly marked invalid while the warning is
  showing.
- **Multiple colors can be targeted at once** (e.g. cyan + blue
  together) -- checking several hue presets simply concatenates their
  hue ranges before matching, reusing `color_detection.py`'s existing
  multi-range matcher rather than adding new logic.
- **Per-ROI minimum-area noise filter**: a detected blob smaller than a
  configurable threshold (mm² if the recording has a physical scale,
  else raw px) is dropped entirely -- excluded from every downstream
  number (area, coverage), not just hidden from display.
- **The combined-fragments overlay draws the true shape, not a convex
  hull.** An early version of the "combine every pad's fragments into
  one outline + one total" display mode used `cv2.convexHull()`, which
  straight-lines across any concave dip in the real shape -- a crescent-
  shaped specular highlight (a very real, very common detected shape on
  a curved reflective pad) would get an outline covering roughly double
  its actual detected area. Fixed by reconstructing the true pixel union
  of every fragment's own exact mask and tracing *that* contour instead;
  confirmed directly against a synthetic crescent that the fix's
  enclosed area (~4400px) matches the true shape (~4360px) where the old
  hull version enclosed ~8460px.
- **Mouse-to-image coordinate mapping accounts for panel letterboxing.**
  The video panels have a hard minimum size; when the computed display
  scale would otherwise produce a smaller pixmap, Qt silently clamps the
  label larger and centers the pixmap inside it (`AlignCenter`) -- a real
  bug (reported as "the box doesn't go where I draw it") where ROI
  rectangles landed offset from the actual drag. Fixed by tracking that
  centering offset explicitly at render time and subtracting it before
  scaling drag coordinates back to image pixels, rather than assuming
  the pixmap always fills the label exactly.

**Analysis runs** (`processing_project.py`) persist Range/Full detection
results separately from the immutable recording, under
`processing_projects/<project_id>/`:

```
processing_projects/<project_id>/
    project.json                        -- ROI definitions, references recording_id
    analysis_runs/<analysis_run_id>/
        run.json                        -- full config/ROI snapshot AS USED, status, stop_reason
        roi_results.jsonl               -- one row per (frame, ROI), references analysis_run_id
        .incomplete                     -- same crash-safety marker as recording_store.py's
```

Every Range/Full execution gets its own `analysis_run_id` and its own
`roi_results.jsonl` -- results from different configs, or reruns, are
never appended into one shared file, which would make "which config
produced this row" ambiguous without cross-referencing every row. The
full config/ROI snapshot is stored once in `run.json`; a cancelled,
crashed, or failed run gets an honest `run.json` (real status, real
stop_reason, however far it actually got) and keeps its `.incomplete`
marker regardless -- only a run that reaches a clean, fully-successful
finish ever loses it, confirmed directly by injecting a real exception
mid-run and checking the marker survives.

### Results tab

Originally planned (see the placeholder note that used to be here, and
the old Stage 9/10 description above) as a live per-circle measurement
view with its own logging. Superseded by the ROI analysis-run system
above, which already produces and persists exactly that kind of
per-frame/per-ROI measurement -- for recorded video, not live. This tab
is now a browser over what that system has actually saved: every
analysis run across every recording's processing project in one table,
with Open Folder, Export CSV (flattens `roi_results.jsonl`'s nested
per-row fields into spreadsheet-friendly columns), Delete, and a jump
back to that recording in Processing.

## Controls (`preview.py` CLI)

- `q` -- quit
- `space` -- pause/resume the preview (camera keeps running, display
  freezes on the last frame)
- `s` -- save the current **raw, unprocessed** frame as a timestamped PNG
- `c` -- open the camera's native settings dialog live, without pausing
  the preview (see above)

## What gets printed on startup

- The OpenCV backend actually in use (`cap.getBackendName()`)
- Requested vs. actual FourCC (pixel format) -- with an explicit warning
  if the requested one wasn't honored
- Requested vs. actual resolution
- Requested FPS vs. the driver-reported FPS (often unreliable for USB
  cameras -- the on-screen preview shows a *measured* FPS instead, based
  on real frame-arrival timestamps)
- On the first successfully received frame: whether it arrived as true
  single-channel grayscale, or as a 3-channel frame with equal R/G/B
  values (grayscale data wrapped in a color format, common for UVC
  drivers), or as genuine color

## Testing it

1. Run `py preview.py --list` and confirm your camera shows up with
   `can_read: True`, a sane resolution, and (if pygrabber is working) its
   real name.
2. Run `py preview.py --index N --list-formats` and confirm you get a
   real (resolution, format, fps range) table, not an error.
3. Run `py preview.py --index N` (your camera's index) and confirm a
   window opens showing a live, responsive image, and that the printed
   FourCC/resolution/fps match what `--list-formats` said was possible.
4. Check the printed startup info matches what you'd expect (resolution,
   backend, frame format).
5. Press `space` -- the image should freeze; press it again -- it should
   resume.
6. Press `s` -- confirm a `frame_<timestamp>.png` appears in `captures/`
   and looks correct when opened.
7. Unplug the camera while the preview is running -- you should see a
   clear "no frame received" warning followed by an error and clean exit
   after a couple of seconds, not a frozen window.
8. Press `c` -- the native settings dialog should open while the preview
   window keeps updating live; move a slider (e.g. Gain or Exposure in
   Camera Control) and confirm the live image visibly changes in
   real time. Press `c` again while it's open -- should print "already
   open," not a second dialog. Close it -- the console should print the
   resulting exposure/gain.
9. Press `q` -- the window should close and the program should exit
   cleanly, printing how many frames were saved (even if you skipped
   closing the settings dialog first).

## Known Windows/OpenCV quirks to expect

- `cv2.VideoCapture` has no built-in open timeout and can hang on a bad
  index -- probing guards against this (see `--probe-timeout` above), but
  if you call `CameraStream` directly in your own code, be aware opening
  an unresponsive index can block indefinitely.
- The MSMF backend (often what `any` resolves to) has poor UVC control
  support for this camera specifically per the vendor's own script --
  that's why `dshow` is the default here, not just generic Windows advice.
- `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT/FPS)` is a request, not a
  guarantee. Always check the "actual" values the program prints on
  startup.
- A monochrome sensor may still arrive as a 3-channel frame if the
  driver reports a color-like UVC format (e.g. YUY2/MJPEG) that OpenCV
  converts to BGR24 -- this is normal and is called out explicitly in
  the startup log, not treated as an error.
