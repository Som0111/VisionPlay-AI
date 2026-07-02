# VisionPlay AI — Plugin Development Guide

## Overview

Every game, utility, or demo in VisionPlay AI is a plugin: a self-contained module under
`src/visionplay/apps/<app_name>/` implementing the `AppPlugin` interface. The `apps/` tree
is flat — an app's category is declared in its manifest (that's what the launcher groups
by), never encoded in the folder path. This guide describes the contract new apps must
follow.

## Standard app folder shape

```
apps/<app_name>/
|-- __init__.py
|-- manifest.py       # AppManifest instance
|-- plugin.py          # AppPlugin implementation (lifecycle glue)
|-- processor.py       # CV/game logic, kept separate from UI for testability
|-- widget.py           # Qt widget(s) specific to this app (overlays, controls)
`-- assets/             # icons, sample images specific to this app
```

## `AppManifest` fields

- `id` — unique, stable string identifier (never reused across apps)
- `name` — display name in the launcher
- `category` — one of the defined app categories (`gesture_games`, `fitness`, `face_ar`, `ai_demos`)
- `version` — the app's own version, independent of the platform version
- `api_version` — which version of the `AppPlugin` contract this app targets; used by the
  registry to reject apps built against an incompatible interface
- `required_backends` — list of inference backends/capabilities this app needs (e.g.
  `["mediapipe.hands"]`, `["onnx"]`); used for capability negotiation in the launcher
- `icon` — path to the app's launcher icon, relative to its `assets/` folder

## `AppPlugin` lifecycle

- `on_load` — called once when the registry discovers and instantiates the plugin; do
  cheap setup only (no camera/model access yet)
- `on_start` — called when the user opens the app from the launcher; acquire camera/model
  resources here
- `on_frame(frame)` — called once per frame **on the pipeline worker thread** (never the Qt
  main thread). The frame arrives with inference results already attached
  (`frame.results` — landmarks/detections from the backends declared in the manifest);
  plugins never run backends themselves. Delegate logic to `processor.py`, never touch Qt
  objects from here, and keep per-frame cost bounded — a slow `on_frame` causes the
  pipeline to drop frames for your app, not queue them
- `on_stop` — release camera/model resources
- `on_unload` — called when the app is removed from the registry (not expected in v1's static
  discovery, but must be implemented for forward compatibility)

## Rules

- `processor.py` must not import anything from `PySide6` — it should be testable by feeding
  it synthetic frames in a plain pytest test, with no display or event loop running.
- `widget.py` is the only place Qt-specific rendering/controls belong.
- No app imports another app's modules. If two apps need the same logic, it belongs in
  `core/` or `vision/`, not copy-pasted or cross-imported.
- Apps must fail gracefully if a `required_backend` isn't available — the launcher greys out
  apps proactively, but `on_start` should still guard against a backend becoming unavailable
  between checks.
- `widget.py` receives per-frame results only via the thread-safe Qt signal the pipeline
  provides — never by calling into the plugin/processor from the UI thread while frames are
  flowing.
- Exceptions escaping `on_frame` are caught by the registry: they're logged, and repeated
  consecutive failures stop the app with an error dialog. Don't rely on this as control
  flow — it exists so one broken app can't crash the shell.

## Adding a new app — checklist

1. Copy `apps/_template/` to `apps/<app_name>/` (flat — no category subfolder).
2. Fill in `manifest.py` — pick a unique `id`, correct `category`, accurate `required_backends`.
3. Implement game/utility logic in `processor.py`, unit-testable headless.
4. Implement `plugin.py` lifecycle methods, delegating logic to `processor.py`.
5. Build the app's UI in `widget.py`.
6. Add unit tests under `tests/unit/apps/<app_name>/`.
7. Verify the app appears correctly in the launcher and is greyed out cleanly if its
   `required_backends` aren't satisfied (test by temporarily disabling the backend).

## Roadmap note

v1 discovery is internal-only (`pkgutil.iter_modules` over `apps/`). A later phase adds
external, pip-installable plugins via `importlib.metadata.entry_points` — when that lands,
this guide's `AppPlugin`/`AppManifest` contract should not need to change, only the discovery
mechanism.
