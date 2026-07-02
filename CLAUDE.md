# VisionPlay AI

Production-quality desktop platform for computer vision games, utilities, and AI demos.
Built with Python, OpenCV, MediaPipe, ONNX Runtime, and PySide6.

## What this is

A single desktop app that hosts many small, self-contained CV "apps" — gesture-controlled
games, fitness/pose utilities, face/AR filters, and general AI vision demos — behind one
launcher/dashboard UI. Each app is a plugin implementing a common interface; the platform
(camera pipeline, inference backends, UI shell) is shared infrastructure the plugins build on.

Windows-only for v1. Core/vision code must stay platform-portable (no OS-specific
assumptions) even though only Windows is packaged right now — macOS/Linux support is a
planned later phase, not a rewrite.

## Architecture (see `docs/architecture.md` for full detail)

Strict layering, dependencies only point downward:

```
ui/  ->  apps/ (plugins)  ->  vision/  ->  core/
                                            ^
                                        services/
```

- `core/` — plugin registry, event bus, config, logging, paths. Never imports from `apps/` or `ui/`.
- `vision/` — camera abstraction, frame pipeline (producer/consumer via worker thread),
  inference backend abstraction (`MediaPipeBackend`, `ONNXBackend`). No Qt imports.
- `apps/` — flat, one subfolder per plugin (game/utility/demo); category lives in the
  manifest, never in the folder path. Each app has `manifest.py`, `plugin.py` (lifecycle
  glue), `processor.py` (pure CV logic, no Qt — must be unit-testable headless), `widget.py`
  (Qt-specific rendering/controls). Underscore-prefixed folders (`_template/`) are skipped
  by plugin discovery.
- `ui/` — PySide6 shell: main window, launcher/dashboard, shared widgets, theming.
- `services/` — settings persistence, opt-in local telemetry, update checker, crash reporter.

## Plugin system (see `docs/plugin-development.md`)

- `AppPlugin` (ABC) defines lifecycle: `on_load`, `on_start`, `on_frame`, `on_stop`, `on_unload`.
  Must stay framework-agnostic — no Qt in the interface itself. This is what keeps the door
  open for a future sandboxed/external plugin loader without a breaking interface change.
- `AppManifest` declares `id`, `name`, `category`, `version`, `api_version`, `required_backends`,
  `icon`. `api_version` exists from day one even though only internal plugins consume it now —
  treat the manifest/plugin contract as a public API, not an implementation detail.
- v1 discovery is internal only (`pkgutil.iter_modules` over `apps/`). External/pip-installable
  plugins via `entry_points` are a later phase — when that lands, only the discovery mechanism
  changes, not `AppPlugin` itself.
- New app checklist: copy `apps/_template/`, fill in `manifest.py`, keep all CV/game logic in
  `processor.py` so it can be tested without a display.

## Vision pipeline rules

- Camera capture, inference, and the plugin's `on_frame` all run on the pipeline worker
  thread, never the Qt main thread. `on_frame` must not touch Qt objects; results reach
  `widget.py` only via the thread-safe Qt signal. Frames flow through `frame_bus` (bounded,
  frame-skipping — slow apps drop frames, they don't queue).
- The pipeline owns inference backends; plugins declare needs via `required_backends` and
  receive results on `frame.results`. Plugins never instantiate backends.
- Backend selection is capability-negotiated: the launcher checks each app's
  `required_backends` against what's actually available and greys out apps it can't run,
  rather than letting them crash at launch.
- v1 is CPU-only — no GPU dependency, no `onnxruntime-gpu`, no CUDA/DirectML. Backend
  constructors still take a `device` param (currently only `{"type": "cpu"}`) and
  `ONNXBackend` requests execution providers as an ordered list rather than hardcoding one, so
  GPU support later is additive (append a provider, add a config option) — never assume the
  reverse and hardcode CPU-specific logic into `vision/inference/`. See `docs/architecture.md`
  § 5 for the full rationale.
- v1 runs exactly one app at a time with exclusive camera access — no multi-consumer
  arbitration.
- A plugin exception must never crash the shell: the registry guards lifecycle calls and
  stops an app after repeated `on_frame` failures.

## Commands

(Fill in once the project is scaffolded with real code — placeholders below.)

```
# run the app
python -m visionplay

# tests
pytest tests/unit
pytest tests/integration

# lint / type-check
ruff check src/
mypy src/

# build Windows package
scripts/build_windows.ps1
```

## Conventions

- No Qt imports in `core/` or `vision/` — those layers must be usable/testable headless.
- No app-to-app imports — plugins are isolated; shared logic goes in `core/` or `vision/`, not
  borrowed from a sibling app.
- Keep per-frame work in `on_frame`/`processor.py` bounded — the pipeline drops frames for
  slow apps rather than queueing, so blocking calls there degrade that app's own experience.
- The event bus is for platform events in Qt-free layers only (`core/`, `vision/`); UI-to-UI
  communication uses native Qt signals, never the bus.
- Large model files are never committed — `model_registry.py` downloads and checksum-verifies
  into a gitignored local cache (`models/`).
- Camera frames never leave the device. Any telemetry is opt-in and metadata-only — state this
  explicitly in user-facing docs since this is a public portfolio project.

## Current phase

Phase 0 — Foundation (repo scaffold, core primitives, camera-to-Qt render pipeline proof,
PyInstaller packaging spike). See `docs/roadmap.md` for the full phase breakdown.

## Do / Don't for future sessions

- Do keep `processor.py` (logic) and `widget.py` (Qt) separate in every app — this is the
  main thing that keeps apps testable and swappable.
- Do add `api_version` handling to any change touching `AppManifest` — it's a public contract.
- Don't add third-party/external plugin loading yet — that's an explicit later phase (Phase 6),
  not an implicit "while I'm here" addition.
- Don't bundle large model binaries into the repo — use `model_registry.py`'s download/cache path.
- Don't introduce platform-specific code in `core/` or `vision/` without a portability shim —
  v1 ships Windows-only, but the layers underneath are meant to outlive that constraint.
