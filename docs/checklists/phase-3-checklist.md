# Phase 3 — Vertical Slice Applications Checklist

Phase 3 builds the seven planned apps on top of the plugin/pipeline/inference framework
proved out in Phases 0–2. `AppPlugin`, `FramePipeline`, `BackendManager`, and the MediaPipe
hand-landmark + ONNX backends are already load-bearing and unchanged in shape
(`docs/architecture.md` §3–5) — this phase fills in real games/utilities/demos, not new
platform abstraction, with one exception: `mediapipe_backend.py` only implements
`MediaPipeTask.HAND_LANDMARKS` today (pose/face are explicitly reserved), so pose and face
support must be added before the apps that need them.

Each milestone below is scoped to fit in a single Claude session, independently testable and
reviewable, and builds on the milestones before it. A milestone is "done" when its
**Done when** criteria all hold. Out of scope for this phase: async threading, telemetry
dashboard, hot-swappable plugins, OCR, semantic segmentation, multi-object tracking, and
analytics/industry-specific apps — all deferred to later roadmap phases.

---

## M3.1 — Vision foundations: pose & face backends + shared gesture toolkit
- [x] `vision/inference/mediapipe_backend.py` (modify) — implement
      `MediaPipeTask.POSE_LANDMARKS` and `MediaPipeTask.FACE_LANDMARKS` alongside the existing
      `HAND_LANDMARKS` body, returning standardized landmark output through the same
      `results.py` value objects used today — no change to the `InferenceBackend` contract.
- [x] `vision/inference/model_catalog.py` (modify) — add `POSE_LANDMARKER` and
      `FACE_LANDMARKER` `ModelSpec`s (download URL + checksum), resolved through the existing
      `ModelRegistry` cache; no model binaries committed to the repo.
- [x] `vision/inference/backend_defaults.py` (modify) — add
      `register_mediapipe_pose_backend`/`register_mediapipe_face_backend` factories exposed as
      `"mediapipe.pose"`/`"mediapipe.face"`, wired into `register_default_backends` alongside
      the existing `"mediapipe.hands"` registration.
- [x] New `vision/gestures/` package — pure, Qt-free helpers shared across apps: fingertip/
      landmark accessors, pinch and finger-count detection, a smoothing filter (One-Euro or
      EMA), and a velocity tracker. No app-to-app imports later; this is the shared home for
      that logic per `docs/architecture.md` §3.
- [x] Unit tests: pose/face backends return standardized output for a fixture frame with a
      detectable subject and empty output for a blank frame, without raising; gesture toolkit
      functions verified against synthetic landmark arrays (pinch true/false thresholds,
      finger-count edge cases, smoothing filter convergence).

**Done when:** `BackendManager` can construct and report availability for `mediapipe.pose` and
`mediapipe.face` the same way it already does for `mediapipe.hands`, and a gesture toolkit
exists in `vision/` that any app's `processor.py` can import — verified by headless unit tests,
with no live camera or Qt required, and no regression to the existing hand-tracking demo.

---

## M3.2 — Air Canvas MVP: gesture drawing core
- [x] Scaffold `apps/air_canvas/` from `apps/_template/` (`required_backends:
      ["mediapipe.hands"]`), establishing the plugin pattern the remaining apps in this phase
      copy.
- [x] `processor.py` — headless canvas state model: stroke list, pinch-to-draw (pen down/up)
      using M3.1's pinch detector, smoothed fingertip path via the shared smoothing filter,
      brush size/color state, eraser mode, clear-all gesture.
- [x] `widget.py` — composited camera + canvas render; toolbar for color/brush/eraser/clear
      with hover-or-pinch selection over toolbar regions.
- [x] Unit tests: stroke building from synthetic landmark sequences, pinch pen-down/up state
      machine, eraser hit-testing — all against `processor.py` with no widget/Qt involved.

**Files expected to change:** new `src/visionplay/apps/air_canvas/{__init__,manifest,plugin,
processor,widget}.py`; `tests/unit/apps/test_air_canvas_processor.py`.

**Done when:** Air Canvas launches from the dashboard, pinch reliably drives pen up/down with
smooth (non-jittery) strokes, brush/color/eraser/clear all work via gesture, and the app starts/
stops cleanly without crashing the shell.

---

## M3.3 — Air Canvas flagship features: undo/redo, layers, shapes, export
- [x] Command-pattern undo/redo stack in `processor.py` covering stroke add, erase, and clear.
- [x] Layer model: add/select/toggle-visibility/reorder, with per-layer compositing in
      `widget.py`.
- [x] Shape recognition: classify a completed stroke as a near-circle/line/rectangle and
      snap-replace it, toggleable on/off.
- [x] Export: save the canvas (with or without camera background) to PNG via a file dialog;
      keyboard shortcuts for undo/redo/save.
- [x] Unit tests: undo/redo invariants (including redo-stack invalidation on new strokes),
      layer compositing order, shape classifier against synthetic stroke fixtures.

**Files expected to change:** `src/visionplay/apps/air_canvas/processor.py`, `widget.py`
(and `plugin.py` if lifecycle hooks need it); `tests/unit/apps/test_air_canvas_*.py`.

**Done when:** Air Canvas supports multi-step undo/redo, multiple independently-toggleable
layers, shape snapping, and PNG export — all covered by headless unit tests on `processor.py` —
and is demo-ready as the flagship app.

---

## M3.4 — Fruit Ninja game
- [ ] Scaffold `apps/fruit_ninja/` (`required_backends: ["mediapipe.hands"]`); `processor.py`
      owns the full game simulation as pure, bounded-per-call logic driven from `on_frame`:
      fruit spawner, parabolic physics, fingertip blade trail with a velocity threshold (using
      M3.1's velocity tracker), segment-vs-circle slice collision, miss/bomb handling, score and
      combo multiplier, lives, game-over/restart states.
- [ ] `widget.py` — render fruits/halves/blade trail/particles and a HUD (score, combo, lives),
      plus a start/restart overlay.
- [ ] Unit tests: physics stepping determinism, collision detection against synthetic blade
      paths, combo scoring rules, and game state transitions — all headless.

**Files expected to change:** new `src/visionplay/apps/fruit_ninja/*.py`; `tests/unit/apps/
test_fruit_ninja_processor.py`.

**Done when:** Fruit Ninja is fully playable end-to-end (slow hand movement doesn't slice, fast
swipes do, combos and bombs work, restart resets cleanly) at a playable frame rate, validating
the pipeline's frame-skipping policy under sustained per-frame game-loop work.

---

## M3.5 — Fitness Rep Counter
- [ ] Scaffold `apps/rep_counter/` (`required_backends: ["mediapipe.pose"]`) — first consumer
      of the M3.1 pose backend.
- [ ] `processor.py` — joint-angle computation from pose landmarks; per-exercise config (bicep
      curl, squat, push-up, shoulder press) as data: joint triple + up/down angle thresholds +
      hysteresis, so a rep counts once per full range of motion, not on noise.
- [ ] Real-time form feedback rules (e.g. "go lower") and per-session stats (reps, sets, tempo).
- [ ] `widget.py` — skeleton overlay, joint-angle arc display, rep counter HUD, exercise
      selector, session summary view.
- [ ] Unit tests: angle math against synthetic pose landmarks, rep state machine against
      synthetic angle sequences (no double-counting, hysteresis holds through noise).

**Files expected to change:** new `src/visionplay/apps/rep_counter/*.py`; `tests/unit/apps/
test_rep_counter_processor.py`.

**Done when:** performing a real set of curls or squats on camera produces a rep count matching
the actual reps performed, partial reps don't count, switching exercises resets state cleanly,
and the pose backend is proven end-to-end through a real app.

---

## M3.6 — AI Object Detection
- [ ] `vision/inference/model_catalog.py` (modify) — add a CPU-friendly detection `ModelSpec`
      (e.g. a YOLOv8n/NanoDet ONNX export, checksum-verified, never committed) and register it
      via `backend_defaults.py` — first real-model consumer of the ONNX path.
- [ ] Detection post-processing (in `vision/` or the app's `processor.py`): letterbox
      preprocessing, output decode, confidence filtering, class-agnostic NMS, COCO label
      mapping.
- [ ] Scaffold `apps/object_detection/`; `processor.py` applies confidence threshold and
      per-class filtering, with optional every-Nth-frame inference to preserve CPU headroom
      under the existing frame-skipping policy.
- [ ] `widget.py` — class-color-stable boxes with labels/confidence, threshold slider, class
      filter list, FPS/inference-latency readout.
- [ ] Unit tests: NMS, box decode, and letterbox coordinate mapping against golden fixture
      arrays — no live model required for these.

**Files expected to change:** `src/visionplay/vision/inference/model_catalog.py`,
`backend_defaults.py`; new `src/visionplay/apps/object_detection/*.py`; `tests/unit/...`.

**Done when:** pointing the camera at real objects produces correct, stable-colored bounding
boxes at an acceptable CPU frame rate, the confidence threshold slider visibly filters
detections, and the model downloads/caches correctly on first run via `ModelRegistry`.

---

## M3.7 — Face Filter (AR)
- [ ] Scaffold `apps/face_filter/` (`required_backends: ["mediapipe.face"]`) — first consumer
      of the M3.1 face backend.
- [ ] `processor.py` — anchor-point extraction (eyes, nose bridge, forehead, chin), face
      scale/roll estimation, landmark smoothing (via the M3.1 toolkit) to prevent overlay
      jitter, alpha-blended sprite placement with rotation/scale.
- [ ] Filter assets: small alpha-PNG sprites shipped in the app folder (not model-registry
      material — these are small, static assets, not downloaded models), plus at least one
      procedural effect (e.g. mesh/mask tint).
- [ ] `widget.py` — filter carousel/selector; multi-face support if the backend surfaces it.
- [ ] Unit tests: anchor math, roll/scale estimation from synthetic landmarks, sprite placement
      bounds-safety when a face is near the frame edge.

**Files expected to change:** new `src/visionplay/apps/face_filter/*.py` and an `assets/`
subfolder; `tests/unit/apps/test_face_filter_processor.py`.

**Done when:** overlays track head scale and roll smoothly across multiple filters without
jitter, tolerate the face moving to the frame edge or leaving/re-entering the frame without
crashing, and the face backend is proven end-to-end through a real app.

---

## M3.8 — Virtual Mouse & Keyboard
- [ ] New `InputController` abstraction (protocol + a concrete OS-input implementation) so
      OS-specific input injection sits behind a portability shim, matching the "no
      platform-specific code without a shim" rule for anything below `apps/`; `processor.py`
      emits abstract intents (`move`, `click`, `drag_start`, …) and never touches OS APIs
      directly.
- [ ] Scaffold `apps/virtual_mouse/` (`required_backends: ["mediapipe.hands"]`): fingertip →
      screen-coordinate mapping with an active-region margin, smoothing via the M3.1 filter,
      pinch = left click, an alternate gesture = right click, pinch-hold = drag, two-finger
      vertical motion = scroll, plus a dwell-click fallback.
- [ ] Virtual keyboard mode: on-screen key grid in `widget.py`, hover-dwell or pinch to type
      via the same `InputController`.
- [ ] Safety: explicit arm/disarm toggle and a kill gesture so the app can never trap the
      user's real cursor.
- [ ] Unit tests: coordinate mapping, gesture-to-intent state machine, dwell timing — all
      against a mock `InputController`, with no real input injected during tests.

**Files expected to change:** new `src/visionplay/apps/virtual_mouse/*.py` (including
`input_controller.py`); dependency addition for OS input (e.g. `pynput`) in `pyproject.toml`;
`tests/unit/apps/test_virtual_mouse_processor.py`.

**Done when:** cursor movement, left/right click, drag, and scroll all work reliably via
gesture against real desktop UI, the virtual keyboard types correctly, and disarming
immediately and completely releases OS input control.

---

## M3.9 — QR/Barcode Scanner + Phase 3 closeout
- [ ] Scaffold `apps/qr_scanner/` with an empty `required_backends` list — verifies the
      launcher/pipeline correctly handle a backend-free plugin.
- [ ] `processor.py` — multi-code detection/decoding per frame (`cv2.QRCodeDetector` and
      `cv2.barcode.BarcodeDetector` for 1-D formats) with result de-duplication and a cooldown
      so a stationary code doesn't spam results.
- [ ] `widget.py` — bounding polygon and decoded-text overlay, scan history list, copy-to-
      clipboard and open-URL actions (URL open gated behind a confirmation prompt).
- [ ] Unit tests: decode against fixture images in `tests/fixtures/`, dedupe/cooldown logic,
      URL-vs-plain-text classification.
- [ ] Closeout: verify all seven apps appear correctly categorized in the launcher with icons,
      and that the capability-negotiation grey-out from Phase 2 still works correctly for the
      apps with `required_backends`; update `docs/roadmap.md` and this repo's `CLAUDE.md`
      "Current phase" note; full `pytest`, `ruff check src/`, `mypy src/` sweep.

**Files expected to change:** new `src/visionplay/apps/qr_scanner/*.py`; new fixture images
under `tests/fixtures/`; `tests/unit/apps/test_qr_scanner_processor.py`; `docs/roadmap.md`;
`CLAUDE.md`.

**Done when:** scanning a real QR code and a printed barcode decodes instantly without duplicate
spam, copy/open actions work, all seven Phase 3 apps launch and stop cleanly in sequence from
the dashboard, and `pytest`/`ruff`/`mypy` are green.

---

## Phase 3 exit criteria
- [ ] All milestones M3.1–M3.9 complete.
- [ ] `mediapipe.pose` and `mediapipe.face` backends run real CPU-only inference alongside the
      existing `mediapipe.hands` backend, with no change in shape to `InferenceBackend`,
      `BackendManager`, or the capability-negotiation contract from Phase 2.
- [ ] A shared, Qt-free gesture toolkit in `vision/gestures/` is reused by Air Canvas, Fruit
      Ninja, and Virtual Mouse & Keyboard — no gesture/smoothing logic duplicated per app, and
      no app-to-app imports anywhere.
- [ ] All seven planned apps (Air Canvas, Fruit Ninja, Fitness Rep Counter, AI Object Detection,
      Face Filter, Virtual Mouse & Keyboard, QR/Barcode Scanner) are launchable from the
      dashboard, correctly categorized, and each keeps CV/game logic in a headless-testable
      `processor.py` separate from `widget.py`.
- [ ] The launcher's capability negotiation correctly greys out any app whose
      `required_backends` isn't satisfied, including the empty-`required_backends` QR scanner
      case.
- [ ] v1 remains CPU-only, Windows-only-but-portable, and no `AppPlugin`/`AppManifest` signature
      changed from Phase 1/2 to accommodate any of the seven apps.
- [ ] No third-party/external plugin loading was introduced (still reserved for a later phase).
- [ ] No large model binaries committed — all new models (pose, face, detection) flow through
      `model_registry.py`'s download/checksum/cache path.
- [ ] `ruff`, `mypy`, and `pytest` (including all new app and gesture-toolkit tests) all green
      in CI on Windows.
- [ ] `core/` and `vision/` still contain zero Qt imports; nothing below `ui/`/`apps/` imports
      upward; `apps/` folders remain flat with no cross-app imports.
- [ ] Tagged as a Phase 3 checkpoint before Phase 4 begins.
