# Phase 0 — Foundation Checklist

Phase 0 proves the skeleton end-to-end: a packaged Windows app that opens a window, pulls
camera frames on a worker thread, and renders them in Qt — plus the core primitives every
later phase builds on. **No app plugins, no inference backends yet** (those are Phases 1–2).

Each milestone below is scoped to fit in a single Claude session. They are ordered by
dependency; check them off in sequence. A milestone is "done" when its **Done when** criteria
all hold.

---

## M0.1 — Project skeleton & tooling
- [ ] `pyproject.toml` with project metadata, `src/visionplay` package, and pinned dependencies
      (PySide6, opencv-python, mediapipe, onnxruntime — CPU wheel, platformdirs, PyYAML).
- [ ] Lockfile via uv (or poetry) committed.
- [ ] Dev tooling configured: `ruff` (lint + format), `mypy` (strict on `core/` and `vision/`),
      `pytest`.
- [ ] `.gitignore` covering `models/`, build artifacts, `__pycache__`, venv.
- [ ] `python -m visionplay` runs and prints a version/banner (no window yet).

**Done when:** `pip install -e .` succeeds in a clean venv, `ruff check src/` and `mypy src/`
pass on the empty package, `python -m visionplay` exits 0.

---

## M0.2 — Core: paths, config, logging
- [ ] `core/paths.py` — `platformdirs`-based resolution for config dir, cache dir, models dir,
      log dir. No hardcoded Windows paths.
- [ ] `core/config.py` — load/save `config.yaml` with per-namespace sections; sane defaults
      created on first run.
- [ ] `core/logging_setup.py` — structured logging to file (in the log dir) + console, with a
      configurable level from config.
- [ ] Unit tests for paths (monkeypatched dirs) and config (round-trip load/save/defaults).

**Done when:** first run creates a config file and log file in the correct platform dirs; unit
tests pass; nothing in `core/` imports Qt or `vision/`.

---

## M0.3 — Core: event bus
- [ ] `core/event_bus.py` — minimal in-process pub/sub for platform events (typed event
      objects, `subscribe`/`publish`). No Qt.
- [ ] Define the initial platform event types as stubs (app started/stopped, backend
      availability changed, camera lost/recovered) — just the dataclasses, no publishers yet.
- [ ] Unit tests: subscribe/publish, multiple subscribers, unsubscribe.

**Done when:** tests pass; the bus is Qt-free and has no dependency on `ui/`.

---

## M0.4 — Vision: Frame type & camera source
- [ ] `vision/pipeline/frame_types.py` — `Frame` dataclass (ndarray + timestamp + metadata +
      empty `results` slot).
- [ ] `vision/camera/camera_source.py` — wrapper over `cv2.VideoCapture` with explicit
      backend selection (MSMF/DirectShow fallback on Windows), open/read/release, and clear
      errors on failure (no silent None-returns).
- [ ] `vision/camera/camera_manager.py` — device enumeration + exclusive open/close (single
      active consumer; no arbitration).
- [ ] Unit tests with a fake/synthetic capture source (no real webcam needed in CI).

**Done when:** camera source opens a real webcam locally and yields frames; enumeration lists
devices; tests using a synthetic source pass in CI without hardware.

---

## M0.5 — Vision: frame pipeline (worker thread) & frame bus
- [ ] `vision/pipeline/frame_bus.py` — bounded queue with frame-skipping/FPS-governor policy;
      producer runs capture on a dedicated worker thread, consumer receives latest frame.
- [ ] Worker thread lifecycle: start/stop cleanly, release camera on stop, survive a capture
      error without crashing the process.
- [ ] Unit tests: bounded-queue drop behavior under a slow consumer; clean start/stop.

**Done when:** a headless test drives synthetic frames through the bus, confirms frames drop
(not queue) under back-pressure, and shuts down without leaking the worker thread.

---

## M0.6 — UI shell: main window + camera view (pipeline end-to-end)
- [ ] `app.py` — `QApplication` bootstrap wiring config/logging.
- [ ] `ui/main_window.py` — main window with an empty launcher placeholder.
- [ ] `ui/widgets/` — a camera-view widget that subscribes to the frame pipeline via a
      **thread-safe Qt signal** and renders frames; an FPS overlay.
- [ ] Camera thread → `frame_bus` → Qt signal → widget render path proven with a live webcam.

**Done when:** `python -m visionplay` opens a window showing the live camera feed with an FPS
counter; closing the window stops the worker thread and releases the camera cleanly. This is
the Phase 0 headline deliverable.

---

## M0.7 — Inference backend interface stub (no models yet)
- [ ] `vision/inference/backend_base.py` — `InferenceBackend` ABC. Constructor takes a
      `DeviceConfig` (v1: `{"type": "cpu"}` only). Methods stubbed/documented, not implemented.
- [ ] `vision/inference/model_registry.py` — interface for format-tagged (`onnx`/`tflite`)
      checksum-verified download+cache; a no-op/local-only implementation is fine for Phase 0.
- [ ] Document (docstring) the ordered-execution-provider contract for the future ONNX backend.

**Done when:** the ABC and registry import cleanly and are covered by a smoke test; no real
model is downloaded or run (that's Phase 2). This exists so Phase 1's plugin interface can
reference backend types without them being empty.

---

## M0.8 — CI skeleton
- [ ] `.github/workflows/ci.yml` — on push/PR: install (from lockfile), `ruff check`,
      `mypy src/`, `pytest` (Windows runner).
- [ ] Tests requiring a webcam are marked and skipped in CI; synthetic-source tests run.

**Done when:** CI is green on a PR; a deliberately introduced lint/type/test failure turns it
red.

---

## M0.9 — PyInstaller packaging spike (highest-risk item — do not defer)
- [ ] `scripts/build_windows.ps1` — PyInstaller `--onedir` build.
- [ ] Resolve MediaPipe/ONNX/OpenCV native DLL + data-file bundling (hooks/`--add-data` as
      needed).
- [ ] Verify the packaged `.exe` launches the M0.6 window **on a clean Windows VM** (no Python,
      no dev tools installed) and shows the live camera feed.

**Done when:** the built app runs end-to-end on a clean VM. If bundling can't be solved this
phase, that's a load-bearing finding — surface it before Phase 1, don't paper over it.

---

## Phase 0 exit criteria
- [ ] All milestones M0.1–M0.9 complete.
- [ ] `python -m visionplay` (and the packaged exe) show a live, FPS-counted camera feed and
      shut down cleanly.
- [ ] `ruff`, `mypy`, and `pytest` all green in CI on Windows.
- [ ] `core/` and `vision/` contain zero Qt imports; nothing below `ui/`/`apps/` imports upward.
- [ ] Tagged as a Phase 0 checkpoint before Phase 1 begins.
