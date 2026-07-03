# VisionPlay AI — Architecture

## 1. Architectural Style

Layered, plugin-oriented desktop app:

```
+-----------------------------------------------+
|  UI Layer (PySide6)                            |  main window, launcher/dashboard,
|                                                 |  shared widgets, theming
+-------------------------------------------------+
|  Apps Layer (plugins)                          |  each game/utility/demo = one
|                                                 |  self-contained plugin
+-------------------------------------------------+
|  Vision/AI Layer                               |  camera abstraction, frame pipeline,
|                                                 |  inference backends (MediaPipe/ONNX)
+-------------------------------------------------+
|  Core/Platform Layer                           |  plugin registry, event bus, config,
|                                                 |  logging, paths
+-------------------------------------------------+
|  Services                                      |  settings, telemetry (opt-in),
|                                                 |  update checker, crash reporter
+-----------------------------------------------+
```

Dependency direction is strictly downward — `apps/` may depend on `vision/` and `core/`, but
`core/` and `vision/` never import from `apps/` or `ui/`. This is what makes the plugin
boundary real rather than aspirational.

## 2. Folder Structure

```
VisionPlay-AI/
|-- CLAUDE.md
|-- README.md
|-- pyproject.toml
|-- LICENSE
|-- .github/workflows/ci.yml
|-- docs/
|   |-- architecture.md
|   |-- plugin-development.md
|   `-- roadmap.md
|-- assets/
|   |-- icons/
|   `-- themes/
|-- src/visionplay/
|   |-- __main__.py
|   |-- app.py                       # QApplication bootstrap
|   |-- core/
|   |   |-- plugin_base.py           # AppPlugin ABC + AppManifest dataclass
|   |   |-- plugin_registry.py       # discovery + registration
|   |   |-- event_bus.py
|   |   |-- config.py                # layered settings
|   |   |-- logging_setup.py
|   |   `-- paths.py                 # platformdirs-based path resolution
|   |-- vision/
|   |   |-- camera/
|   |   |   |-- camera_source.py     # abstraction over cv2.VideoCapture
|   |   |   `-- camera_manager.py    # device enumeration, arbitration
|   |   |-- pipeline/
|   |   |   |-- frame_bus.py         # producer/consumer, worker thread
|   |   |   `-- frame_types.py       # Frame dataclass
|   |   `-- inference/
|   |       |-- backend_base.py      # InferenceBackend ABC
|   |       |-- mediapipe_backend.py
|   |       |-- onnx_backend.py
|   |       `-- model_registry.py    # model metadata, download/cache
|   |-- ui/
|   |   |-- main_window.py
|   |   |-- launcher/                # app gallery / dashboard
|   |   |-- widgets/                 # camera view, fps overlay, settings panel
|   |   |-- theme/                   # qss + theme manager
|   |   `-- dialogs/
|   |-- apps/                        # each subfolder = one plugin (flat — category
|   |   |                            #   lives in the manifest, not the folder path)
|   |   |-- _template/               # scaffold for new apps; underscore prefix =
|   |   |                            #   skipped by plugin discovery
|   |   `-- <app_name>/              # e.g. squat_counter/, air_hockey/
|   `-- services/
|       |-- settings_service.py
|       |-- telemetry_service.py     # local-only, opt-in
|       |-- update_checker.py
|       `-- crash_reporter.py
|-- tests/{unit,integration,fixtures}/
|-- scripts/{build_windows.ps1,package_release.ps1}
`-- models/                          # gitignored local model cache
```

The `apps/` tree is deliberately flat: an app's category is declared once, in its manifest,
and the launcher groups by that field. Encoding category in the folder path as well would
create a second source of truth that can silently diverge from the manifest.

Every app plugin follows the same internal shape (e.g. `apps/squat_counter/`):
`manifest.py`, `plugin.py`, `processor.py` (pure CV/logic, no Qt — testable), `widget.py`
(Qt-specific parts), `assets/`. Separating `processor.py` from `widget.py` is the single most
important convention here — it's what lets core logic be unit-tested without a display or
event loop.

## 3. Plugin System (Hybrid Design)

- **`AppPlugin` ABC** — lifecycle hooks: `on_load`, `on_start`, `on_frame(frame) -> Frame`,
  `on_stop`, `on_unload`. Framework-agnostic (no Qt imports allowed in the interface itself)
  so a future external/sandboxed plugin isn't blocked by a UI-coupled contract.
- **`AppManifest`** — dataclass declaring `id`, `name`, `category`, `version`, `api_version`,
  `required_backends` (e.g. `["mediapipe.pose"]` or `["onnx"]`), `icon`.
- **Discovery (v1)** — `pkgutil.iter_modules` over `apps/`, internal only; modules whose
  name starts with `_` (e.g. `_template`) are skipped. The discovery mechanism is
  intentionally the only part that changes when external plugins land in Phase 6 (swapped
  for `importlib.metadata.entry_points`) — the `AppPlugin` interface itself does not change.
  No `plugins_external/` folder exists until then; reserving the name here is enough.
- **Failure containment** — the registry wraps every lifecycle call (especially `on_frame`)
  in a guard: an exception is logged with the app id, and N consecutive `on_frame` failures
  stop the app and surface an error dialog. A misbehaving plugin must never crash the shell.
  (Process-level isolation is the Phase-6 upgrade of this same policy.)
- **Capability negotiation** — launcher checks each manifest's `required_backends` against
  what's actually available (installed/GPU-capable) and greys out apps that can't run rather
  than letting them crash at launch.
- **`api_version`** in the manifest from day one — since the plan is "external-ready later,"
  this field is what lets a future plugin loader reject incompatible third-party plugins
  instead of discovering the need for versioning after the fact.

## 4. Vision Pipeline & Concurrency

Camera capture and inference must never run on the Qt main thread.

**Per-frame data flow** (all on one dedicated worker thread):

```
camera_source (capture)
  -> inference backends declared in the active app's required_backends
       (results attached to Frame.results — landmarks, detections, ...)
  -> active plugin's on_frame(frame)  (game/app logic via processor.py)
  -> frame_bus publishes -> thread-safe Qt signal -> widget renders
```

**Threading contract** (explicit, because this is the most bug-prone seam):

- `on_frame` executes **on the pipeline worker thread**, never the Qt main thread. It must
  not touch Qt objects; the app's `widget.py` receives results only via the Qt signal.
- Plugins do not instantiate inference backends themselves. The pipeline owns backend
  lifecycles, runs the backends the active app declared in its manifest, and delivers
  results on the `Frame`. This is what keeps backends shared, swappable, and warm across
  app switches.
- The `frame_bus` is bounded with a frame-skipping/FPS-governor policy: a slow `on_frame`
  drops frames rather than backing up the queue and freezing the UI.

**v1 simplifying constraint**: exactly one app is active at a time. Camera acquisition is
exclusive; `camera_manager.py` does device enumeration and exclusive hand-off on app
switch, not multi-consumer arbitration. Concurrent apps sharing a camera feed is out of
scope until a concrete need appears.

**Event bus scope**: `event_bus.py` exists only because `core/` and `vision/` are Qt-free
and therefore cannot use signals/slots. It stays a minimal in-process pub/sub for
platform-level events (app started/stopped, backend availability changed, camera
lost/recovered). The UI layer uses native Qt signals; do not route UI-to-UI communication
through the bus, and do not grow it into a general message broker.

## 5. Inference Backend Abstraction

`InferenceBackend` ABC with `MediaPipeBackend` (hand/pose/face landmarks) and `ONNXBackend`
(custom models — e.g. YOLO for the AI-demos category). **Implemented as of Phase 2**: both
backends run real CPU inference (`MediaPipeBackend` builds a MediaPipe Tasks graph for hand
landmarks; `ONNXBackend` runs an `onnxruntime.InferenceSession`), and return **standardized
result objects** (`results.py` — `HandLandmarkResult`, `TensorOutput`, ...) so plugin code
never imports MediaPipe/`onnxruntime` types. A `BackendManager` (`backend_manager.py`) owns
instances by name via registered factory/probe pairs (`backend_defaults.py`), warm-caches them
across app switches, and answers the launcher's availability queries; the concrete model list
lives in `model_catalog.py`. **v1 ships CPU-only** — no GPU dependency, no CUDA/DirectML
package, no GPU-specific code path anywhere in `vision/`. GPU support is deferred, not
designed away: the abstraction is shaped so enabling it later is additive.

**How the abstraction stays GPU-ready without adding GPU code in v1:**

- Every backend constructor takes a `device` parameter (`DeviceConfig` — currently just
  `{"type": "cpu"}`) instead of hardcoding CPU internally. Call sites never branch on device;
  they pass through whatever `config.yaml` resolves to.
- `ONNXBackend` requests execution providers as an ordered list (`onnxruntime`'s native
  mechanism) rather than picking one implicitly. v1 populates that list with
  `["CPUExecutionProvider"]` only. Adding GPU later means appending
  `"DmlExecutionProvider"`/`"CUDAExecutionProvider"` ahead of the CPU entry with graceful
  fallback — a config and packaging change, not a code path change, since `onnxruntime`
  already no-ops to the next provider in the list if a preferred one isn't available.
- `MediaPipeBackend` takes the same shape: MediaPipe's Tasks API accepts a `delegate`
  (`CPU`/`GPU`) per task; v1 always passes `CPU`, but the parameter already exists on the
  call, so flipping it later doesn't touch the surrounding pipeline code.
- `model_registry.py` tags cached models by **format** (`onnx`, `tflite`, `task`), never by
  device —
  device selection is a runtime backend concern, not a model-catalog concern, so the registry
  needs no changes when GPU support lands.
- The v1 dependency set installs CPU-only packages (`onnxruntime`, not `onnxruntime-gpu`;
  stock `mediapipe`). Enabling GPU later is a packaging/build change (swap the wheel, add the
  driver/runtime dependency) layered on top of an interface that already expects it — not a
  redesign of `vision/inference/`.
- Capability negotiation (§3) already greys out apps whose `required_backends` can't be
  satisfied — the same mechanism that handles "MediaPipe not installed" today is what will
  grey out a GPU-only app on a machine without a GPU later, with no new negotiation logic
  needed.

`model_registry.py` also handles checksum-verified download-on-first-use and local caching so
the repo itself doesn't carry large binary model files.

## 6. Config, Data & Distribution

- Settings via `platformdirs` (even though v1 is Windows-only, this keeps the "design
  portable" constraint real rather than nominal) — `config.yaml` with per-app namespaced
  sections.
- Packaging: PyInstaller `--onedir` (not `--onefile` — MediaPipe/ONNX native DLL loading is
  unreliable inside a single-file bundle). This should be spiked early (Phase 1), not left to
  the end, since it's a known source of last-mile surprises.
- No cloud sync, no auto-update in v1 — update checker only notifies.

## 7. Risks & Recommendations

- **MediaPipe/ONNX/protobuf dependency conflicts** — these libraries have a real history of
  clashing transitive pins. Use a lockfile (uv or poetry) and test upgrades in isolation, not
  inline with feature work.
- **PyInstaller + native binaries** — budget the packaging spike into Phase 0/1, not Phase 5,
  or you'll discover bundling breakage right before a release deadline.
- **Plugin interface freeze** — since external plugins are explicitly deferred but planned,
  treat `AppPlugin`/`AppManifest` as a public API from the start (hence `api_version` now, not
  later).
- **Camera backend variability** (DirectShow vs MSMF on Windows, resolution support varies by
  webcam) — `camera_source.py` needs explicit fallback and user-facing errors, not silent
  failure.
- **Model licensing** — some YOLO variants are GPL/AGPL, which is incompatible with the
  project's MIT license; verify each bundled/downloaded model's license before shipping it,
  and prefer MIT/Apache-2.0-licensed model weights where a choice exists.
- **Privacy framing** — this is a webcam app; state explicitly in the README that frames
  never leave the device and any telemetry is opt-in/metadata-only. Matters more than usual
  since this is a public portfolio piece.
- **CPU-only performance ceiling** — some AI-demo candidates (e.g. larger YOLO variants) may
  not hit acceptable FPS on CPU alone. Validate target FPS on CPU during Phase 3's vertical
  slice before committing to a specific model size for that category; the GPU-ready backend
  design (§5) is the escape hatch if a CPU ceiling is hit, but it isn't free — it still
  requires the Phase-6 packaging work to actually ship a GPU build.

## 8. Key Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| Target platform (v1) | Windows-only, portable core | Matches dev environment now, avoids lock-in later |
| Plugin depth | Hybrid — internal now, external-ready | Full extensibility without premature complexity (sandboxing, entry_points) |
| Inference backend | MediaPipe + ONNX Runtime, CPU-only for v1 | Covers landmark tracking and custom/detection models across all four categories without a GPU dependency; backend constructors accept a `device` param so GPU is additive later, not a redesign (§5) |
| License | MIT | Portfolio project; permissive by default |
| App categories (v1) | Gesture games, fitness/pose, face/AR, AI demos | Broad enough to stress-test the plugin abstraction across different workload shapes |
| `apps/` layout | Flat, one folder per app | Category lives in the manifest only; folder-path categories would be a second source of truth |
| `on_frame` execution | Pipeline worker thread, no Qt access | Removes the main ambiguity that causes threading bugs; UI gets results via signal only |
| Backend ownership | Pipeline owns backends, plugins declare needs | Keeps backends shared/warm across app switches; plugins never construct backends |
| Concurrency (v1) | One active app, exclusive camera | Multi-consumer camera arbitration deferred until a concrete need exists |
| Plugin failures | Registry-level guard, N-strike stop | Shell must survive any plugin exception; process isolation deferred to Phase 6 |
