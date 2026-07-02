# Phase 1 — Plugin Skeleton Checklist

Phase 1 proves the plugin boundary end-to-end: a registry that discovers and safely runs
`AppPlugin`s, a launcher UI that lists them, and one trivial hand-tracking "hello world" app
driven through the real frame pipeline. **No real inference and no capability negotiation
yet** — `MediaPipeBackend`/`ONNXBackend` stay the Phase 0 stubs (`NotImplementedError` on
`infer()`); real backend bodies, the model registry, and launcher capability negotiation
against actual backend availability are Phase 2 (`docs/roadmap.md`, `docs/architecture.md`
§5). The hello-world app in this phase proves plugin/pipeline *wiring*, not landmark
accuracy — it runs with `frame.results` empty/stubbed rather than real MediaPipe output.

Each milestone below is scoped to fit in a single Claude session. They are ordered by
dependency; check them off in sequence. A milestone is "done" when its **Done when** criteria
all hold.

---

## M1.1 — Core: `AppManifest` & `AppPlugin` contract
- [ ] `core/plugin_base.py` — `AppManifest` frozen dataclass: `id`, `name`, `category`,
      `version`, `api_version`, `required_backends`, `icon` (`docs/plugin-development.md`).
- [ ] `core/plugin_base.py` — `AppPlugin` ABC: `on_load`, `on_start`, `on_frame(frame) -> Frame`,
      `on_stop`, `on_unload`. No Qt imports in the module (`docs/architecture.md` §3).
- [ ] `category` is validated against the four defined categories (`gesture_games`, `fitness`,
      `face_ar`, `ai_demos`) — an unrecognized category is a construction-time error, not a
      silent launcher miscategorization.
- [ ] Docstrings treat this module as a public API surface (per CLAUDE.md's `api_version`
      guidance) — `api_version` is documented as the field a future external-plugin loader
      uses to reject incompatible plugins, even though nothing checks it yet in this milestone.
- [ ] Unit tests: `AppManifest` construction/validation (rejects unknown category), `AppPlugin`
      cannot be instantiated without implementing all five lifecycle methods.

**Done when:** `AppManifest`/`AppPlugin` import cleanly with zero Qt/`apps/`/`ui/`
dependencies; unit tests pass; the module docstring states the interface-freeze rationale from
`docs/architecture.md` §7.

---

## M1.2 — Core: plugin registry (discovery + failure containment)
- [ ] `core/plugin_registry.py` — `PluginRegistry` class: discovers apps via
      `pkgutil.iter_modules` over `apps/`, skipping modules whose name starts with `_`
      (`_template`).
- [ ] Registry instantiates each discovered plugin's `AppManifest` + `AppPlugin`, calls
      `on_load` once per discovery, and rejects/logs (does not crash) a plugin whose
      `api_version` the registry doesn't support.
- [ ] Lifecycle guard: every call into a plugin (`on_start`, `on_frame`, `on_stop`,
      `on_unload`) is wrapped so an exception is caught, logged with the app `id`, and does not
      propagate. `on_frame` failures are counted per-app; N consecutive failures stop that app
      and the registry publishes `GameStopEvent(app_id, reason="error")` on the `EventBus`
      (`core/events.py`, already defined in Phase 0).
- [ ] Registry enforces **one active app at a time**: starting an app while another is active
      stops the previous one first (`on_stop` then `on_unload`'s app-switch equivalent is out
      of scope — only start/stop, not unload, happens on a switch).
- [ ] Registry publishes `GameStartEvent`/`GameStopEvent` (existing dataclasses in
      `core/events.py`) around `on_start`/`on_stop`.
- [ ] Unit tests: discovery skips underscore-prefixed packages; a plugin raising in `on_load`
      is logged and excluded, not fatal to the registry; a plugin raising repeatedly in
      `on_frame` is stopped after the failure threshold; starting app B while app A is active
      stops app A first; events are published in the right order.

**Done when:** the registry discovers a synthetic test-fixture plugin package, runs its full
lifecycle, survives an injected exception at every lifecycle stage without raising out of the
registry, and enforces single-active-app exclusivity — all covered by unit tests with no real
camera or Qt involved.

---

## M1.3 — Apps: `_template` scaffold
- [ ] `apps/_template/__init__.py`
- [ ] `apps/_template/manifest.py` — example `AppManifest` instance with placeholder `id`
      (clearly marked as a template, not a real registrable app).
- [ ] `apps/_template/plugin.py` — `AppPlugin` implementation with all five lifecycle methods
      stubbed and commented to explain what belongs in each, delegating to `processor.py`.
- [ ] `apps/_template/processor.py` — empty pure-logic class with no `PySide6` import, ready
      to be unit-tested headless.
- [ ] `apps/_template/widget.py` — minimal `QWidget` subclass stub showing where per-app
      Qt/rendering code goes, receiving results only via a Qt signal (no direct
      plugin/processor calls from the UI thread).
- [ ] `apps/_template/assets/` — empty directory (`.gitkeep`) for icons/sample images.
- [ ] Confirm `pkgutil.iter_modules` discovery (M1.2) skips this folder because of the
      underscore prefix — add a regression test if not already covered by M1.2's tests.

**Done when:** `apps/_template/` matches the folder shape in `docs/plugin-development.md`
exactly, is importable with no errors, and is confirmed (by test) to be excluded from
registry discovery.

---

## M1.4 — Pipeline integration: active-plugin execution on the worker thread
- [ ] `vision/pipeline/frame_bus.py` (modify) — `FramePipeline` gains a way to run the
      registry's currently active plugin's `on_frame(frame)` once per captured frame, on the
      existing capture worker thread, **after** frame capture and before publishing to the bus
      (`docs/architecture.md` §4 data-flow: capture → backends → `on_frame` → publish). Since
      real backends aren't implemented yet (Phase 2), the backend step is a no-op and
      `frame.results` reaches `on_frame` empty — this milestone wires the *seam*, not inference.
- [ ] `on_frame`'s return value (a `Frame`) is what gets published to the bus, so a plugin can
      annotate/pass through the frame for its own `widget.py` to render via the signal bridge.
- [ ] No app is active by default: with no active plugin, the pipeline behaves exactly as it
      does today (M0.5/M0.6 passthrough), preserving existing camera-view behavior.
- [ ] A plugin exception inside this call path is caught by the registry guard from M1.2, not
      by the pipeline itself — the pipeline must not need its own duplicate try/except policy.
- [ ] Unit tests: with a synthetic capture source and a fixture plugin, verify `on_frame` is
      invoked once per frame, on the worker thread (not the calling/test thread), with no
      active plugin the frame passes through unmodified, and a raising plugin doesn't stop the
      pipeline (only the app, per M1.2's failure containment).

**Done when:** a headless test drives synthetic frames through `FramePipeline` with a fixture
plugin set active, confirms `on_frame` ran on the worker thread for each frame, and confirms
existing no-active-plugin behavior (M0.5/M0.6) is unchanged.

---

## M1.5 — UI: launcher/dashboard widget
- [ ] `ui/launcher/launcher_widget.py` — `QWidget` listing apps from the `PluginRegistry`
      (name, category, icon), grouped/filterable by `category`.
- [ ] Selecting an app emits a Qt signal (e.g. `appLaunchRequested(str)` carrying the app
      `id`) — the launcher does not call into the registry's start/stop directly from a
      button handler in a way that couples it to pipeline internals; it signals intent, the
      wiring in M1.6 acts on it.
- [ ] All apps render as enabled/launchable in this milestone — greying out apps whose
      `required_backends` aren't satisfied is explicitly **Phase 2** (`docs/roadmap.md`:
      "capability negotiation in launcher" is a Phase 2 line item), not part of this widget yet.
- [ ] `ui/launcher/__init__.py` if not already present.
- [ ] Unit tests (Qt widget tests per the pattern in `tests/unit/test_camera_view.py`/
      `test_main_window.py`): launcher populates from a fixture registry with N apps, emits the
      launch signal with the correct app `id` on selection, and updates if the registry's app
      list changes.

**Done when:** the launcher widget renders a fixture set of apps grouped by category and emits
a correctly-identified launch signal on selection, verified by unit tests with no real camera
or registry running.

---

## M1.6 — UI: wire launcher into main window + app start/stop flow
- [ ] `ui/main_window.py` (modify) — replace the M0.6 empty launcher placeholder
      (`_build_launcher_placeholder`) with the real `LauncherWidget` from M1.5.
- [ ] `app.py` (modify) — on `appLaunchRequested`, tell the `PluginRegistry` to start that app
      (stopping any previously active app per M1.2's exclusivity rule) and tell the
      `FramePipeline` which plugin is now active for M1.4's `on_frame` hook; on window close
      (`MainWindow.closing`, existing signal), stop the active app before tearing down the
      pipeline.
- [ ] The active app's `widget.py` overlay/controls are shown alongside (or in place of) the
      existing `CameraView` per-frame render path — reusing the existing thread-safe Qt-signal
      bridge (`ui/widgets/frame_bridge.py`) rather than adding a second frame-delivery
      mechanism.
- [ ] Unit tests: launching an app from the launcher starts it in the registry and activates it
      in the pipeline; closing the window stops the active app cleanly (extends existing
      `tests/unit/test_main_window.py`/`test_app.py` coverage).

**Done when:** `python -m visionplay` shows the real launcher panel instead of the Phase 0
placeholder, selecting an app starts it and its `on_frame` runs per-frame on the worker
thread, and closing the window stops the app and releases the camera cleanly — no regression
to the M0.6 headline behavior when no app is selected.

---

## M1.7 — Apps: hand-tracking "hello world" (pipeline wiring proof)
- [ ] `apps/hand_tracking_demo/` (or similarly named, non-underscore folder so discovery picks
      it up) — `__init__.py`, `manifest.py` (`required_backends=["mediapipe.hands"]`,
      declaring the dependency for when Phase 2 makes it real), `plugin.py`, `processor.py`,
      `widget.py`, `assets/`.
- [ ] `processor.py` reads `frame.results.get("mediapipe.hands")` defensively — in this phase
      it is always absent/`None` (no backend runs yet per M1.4's scope), so the processor must
      handle "no results" as the normal case, not an error. This is intentionally forward-
      compatible with Phase 2 populating that key for real.
- [ ] `widget.py` renders the passthrough camera frame plus a placeholder overlay (e.g. "hand
      tracking data not yet available — Phase 2") confirming the render path works, not real
      landmarks.
- [ ] This app is the concrete fixture that exercises M1.1–M1.6 together: registry discovery,
      `on_load`/`on_start`/`on_frame`/`on_stop`, launcher listing, and pipeline `on_frame`
      wiring, without depending on any Phase 2 work.
- [ ] Unit tests under `tests/unit/apps/hand_tracking_demo/`: `processor.py` handles both an
      empty `frame.results` and a synthetic populated one without raising (future-proofing for
      Phase 2), fully headless.

**Done when:** launching "hand-tracking hello world" from the launcher runs it end-to-end
through the real frame pipeline (camera → `on_frame` → render) with a live webcam, using no
real inference; this is the Phase 1 headline deliverable, analogous to M0.6 in Phase 0.

---

## Phase 1 exit criteria
- [ ] All milestones M1.1–M1.7 complete.
- [ ] `PluginRegistry` discovers apps from `apps/` (skipping `_template`), runs the full
      `AppPlugin` lifecycle, and survives a plugin exception at any lifecycle stage without
      crashing the shell.
- [ ] The launcher UI lists discovered apps grouped by category and can start/stop the active
      one; exactly one app is active at a time.
- [ ] The hand-tracking "hello world" app runs live through the real pipeline
      (camera → `on_frame` → Qt-signal render), proving the plugin/pipeline seam end-to-end
      with no real inference wired in yet.
- [ ] `ruff`, `mypy`, and `pytest` all green in CI on Windows.
- [ ] No `required_backends` capability negotiation, no real `MediaPipeBackend`/`ONNXBackend`
      inference bodies, and no `model_registry.py` download/cache logic were added — that
      scope stays in Phase 2.
- [ ] `core/` and `vision/` still contain zero Qt imports; nothing below `ui/`/`apps/` imports
      upward; `apps/` folders remain flat with no cross-app imports.
- [ ] Tagged as a Phase 1 checkpoint before Phase 2 begins.
