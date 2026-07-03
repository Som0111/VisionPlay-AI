# Phase 2 — Inference Backends Checklist

Phase 2 replaces the Phase 1 inference stubs with a complete, CPU-only inference backend
system while preserving the plugin architecture proved in Phase 1 end-to-end. `InferenceBackend`,
`DeviceConfig`, `ModelRegistry`, `MediaPipeBackend`, and `ONNXBackend` already exist as
lifecycle-complete stubs from Phase 0 (`vision/inference/backend_base.py`, `device.py`,
`model_registry.py`, `mediapipe_backend.py`, `onnx_backend.py`) — their constructors, the
`load()`/`infer()`/`unload()` contract, the ordered-execution-provider shape, and the
format-tagged model cache are already final (`docs/architecture.md` §5). **This phase fills in
real bodies and wires them into the pipeline; it does not redesign the abstraction.** No
gesture recognition or game logic is in scope here — the hand-tracking app gains real
landmarks to render, not new behavior (`docs/roadmap.md`: that's Phase 3).

Each milestone below is scoped to fit in a single Claude session. They are ordered by
dependency; check them off in sequence. A milestone is "done" when its **Done when** criteria
all hold.

---

## M2.1 — Backend architecture: manager, registration, configuration, capability
- [x] `vision/inference/backend_manager.py` — `BackendManager`: constructs and owns
      `InferenceBackend` instances by name (e.g. `"mediapipe.hands"`, `"onnx.yolo_nano"`),
      keeping backends **shared and warm across app switches** rather than reloading per
      launch (`docs/architecture.md` §4 — "Backend ownership: Pipeline owns backends, plugins
      declare needs"). No changes to `InferenceBackend`/`DeviceConfig` themselves — this is a
      new layer above the existing Phase 0 abstraction, not a replacement of it.
- [x] `BackendManager` registration: a factory mapping from backend name to a constructor
      closure (e.g. `MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, device)`,
      `ONNXBackend(spec, model_path, device)`), populated at startup so the manager never
      hardcodes an `if/elif` chain that grows with every new backend.
- [x] Configuration: extend `core/config.py`'s per-namespace sections with an `inference`
      namespace — `device` (`{"type": "cpu"}`, resolved via `DeviceConfig.from_mapping`) and
      model cache overrides. Reads the existing config round-trip machinery from Phase 0; does
      not introduce a second config file.
- [x] Capability representation: a way to ask "is backend X available right now" — checks
      that the backend's runtime dependency imports (MediaPipe/`onnxruntime`) and, for ONNX
      backends, that the required model is registered — without loading the backend or
      touching the camera. This is the primitive M2.3's launcher negotiation calls; it does
      not itself grey out UI.
- [x] `BackendManager` never gets called by plugins directly — only the pipeline (M2.3)
      constructs and queries it, preserving the "plugins never instantiate backends" rule from
      `docs/architecture.md` §3/§4.
- [x] Unit tests: registering a backend factory and resolving it by name; requesting an
      unregistered name raises a clear `InferenceError`-style failure, not a `KeyError` leak;
      availability check reports `False` for a backend whose dependency/model isn't present
      without raising; two requests for the same backend name return the same warm instance
      (no duplicate load).

**Done when:** `BackendManager` constructs, shares, and reports availability for backend
instances purely from configuration and registered factories — verified by unit tests with no
real MediaPipe/ONNX runtime, real camera, or Qt involved.

---

## M2.2 — MediaPipe & ONNX backends: real bodies, model registry, standardized outputs
- [x] `vision/inference/mediapipe_backend.py` (modify) — `MediaPipeBackend.load()` builds the
      real MediaPipe Tasks graph for `MediaPipeTask.HAND_LANDMARKS` using the `delegate`
      property already derived from `DeviceConfig` (`"CPU"` only, v1); `infer()` runs the graph
      on `frame.image` and returns real landmark output instead of raising
      `NotImplementedError`. `unload()` releases the graph.
- [x] `vision/inference/onnx_backend.py` (modify) — `ONNXBackend.load()` constructs
      `onnxruntime.InferenceSession(self.model_path, providers=self.providers)` (the ordered
      `("CPUExecutionProvider",)` list is already correct from Phase 0 — no provider-list
      changes needed for CPU-only v1); `infer()` runs `session.run(...)` and returns real
      output instead of raising `NotImplementedError`.
- [x] `vision/inference/model_registry.py` (modify) — implement a real `ModelDownloader`
      (HTTP fetch to a temp path; the registry's atomic-rename-after-checksum flow already
      exists and is untouched) and register at least one real `ModelSpec` so `ensure()` is
      exercised end-to-end against an actual download, not just the existing fake-downloader
      unit tests.
- [x] Model loading/caching: confirm `ModelRegistry.ensure()`/`is_cached()` are called from
      `BackendManager` (M2.1) at backend-construction time for ONNX backends — the model is
      resolved to a verified local path before `ONNXBackend.load()` ever runs, never inside
      `infer()`.
- [x] Standardized inference outputs: a small set of result value objects (e.g. hand/pose
      landmark points, detection boxes+scores+labels) that `MediaPipeBackend`/`ONNXBackend`
      return instead of raw library-native structures — so `frame.results["mediapipe.hands"]`
      has a stable shape a plugin's `processor.py` can consume without importing MediaPipe or
      `onnxruntime` types directly.
- [x] Unit tests: `MediaPipeBackend.infer()` returns the standardized landmark output for a
      synthetic frame with a detectable hand (fixture image) and an empty/no-detection result
      for a blank frame, without raising; `ONNXBackend.infer()` runs a small real ONNX model
      against a synthetic frame and returns standardized output; `ModelRegistry.ensure()`
      against a fake HTTP downloader still passes (regression on Phase 0 tests) and a checksum
      mismatch from the new real downloader is rejected the same way.

**Done when:** both backends load a real runtime, run inference on a real or fixture frame,
and return standardized (not library-native) output — verified by headless unit tests using a
fixture image, with no live camera or Qt required, and no regression to the existing
`ModelRegistry` checksum/caching tests.

---

## M2.3 — Pipeline & launcher integration
- [x] `vision/pipeline/frame_bus.py` (modify) — replace the M1.4 no-op backend step: for the
      active plugin's `required_backends`, resolve each name through `BackendManager` (M2.1)
      and run `infer()` on the worker thread, **before** the active plugin's `on_frame` and
      after capture (`docs/architecture.md` §4 data-flow: capture → backends → `on_frame` →
      publish). This is the same seam M1.4 wired with a no-op; only the no-op is replaced.
- [x] `frame.results` is populated in place, keyed by each backend's `name` property (e.g.
      `frame.results["mediapipe.hands"]`), matching what `apps/hand_tracking_demo/processor.py`
      already reads defensively since M1.7 — no change to that app's read pattern, only to
      whether the key is present.
- [x] `FramePipeline` integration: backends declared by the active app are loaded on app start
      (via `BackendManager`) and stay warm across `on_frame` calls; they are only unloaded on
      an app switch away from every app that needs them (or process shutdown) — not reloaded
      per frame.
- [x] Capability negotiation: `ui/launcher/launcher_widget.py` (modify) — an app's
      `required_backends` are checked against `BackendManager`'s availability check (M2.1) and
      unsatisfied apps render greyed-out/disabled, replacing the "all apps render as
      enabled/launchable" behavior explicitly deferred from M1.5. Selecting a greyed-out app
      does not emit `appLaunchRequested`.
- [x] Preserve existing plugin API compatibility: no changes to `AppPlugin`, `AppManifest`, or
      the `on_frame(frame) -> Frame` signature from Phase 1 — plugins still never instantiate
      backends themselves, and `apps/hand_tracking_demo` and `apps/_template` need no code
      changes to keep working under real inference.
- [x] A backend `load()`/`infer()` failure is contained the same way plugin exceptions are
      (M1.2's registry guard) — a backend error surfaces as a user-presentable `InferenceError`
      and stops/greys out the affected app, never crashes the shell or the pipeline.
- [x] Unit/integration tests: with a fixture plugin declaring a real backend, `frame.results`
      is populated before `on_frame` runs; an app switch stops backends no longer needed by any
      active app and leaves shared ones warm; the launcher greys out a fixture app whose
      `required_backends` includes an unavailable name and still lists a satisfied app as
      launchable; a backend raising `InferenceError` during `infer()` is contained without
      crashing the pipeline (extends the M1.2/M1.4 failure-containment tests with a real
      backend failure instead of a plugin failure).

**Done when:** launching the hand-tracking demo app from the launcher runs real MediaPipe hand
landmark inference on the worker thread, `frame.results["mediapipe.hands"]` reaches
`processor.py`/`widget.py` with real data, an app whose `required_backends` can't be satisfied
is greyed out in the launcher instead of crashing on launch, and no `AppPlugin`/`AppManifest`
call site needed to change.

---

## M2.4 — Optimization, testing & documentation
- [x] Performance: confirm per-frame backend cost stays bounded under the existing
      frame-skipping policy (`docs/architecture.md` §4) — a slow backend causes the pipeline to
      drop frames for the active app, not back up the bus or block other apps; backends stay
      warm across frames (no per-frame re-`load()`), profiled against a live webcam feed, not
      just fixture images.
- [x] Error handling: every backend failure path (missing model file, checksum mismatch,
      runtime import failure, mid-stream `infer()` exception) raises a user-presentable
      `InferenceError`/`ModelRegistryError` with the backend/model name and reason, never a
      bare library exception leaking up through `BackendManager` or the pipeline.
- [x] Unit & integration tests: full headless coverage for `BackendManager`, both real backend
      bodies, the real `ModelDownloader`, and capability negotiation (per M2.1–M2.3); at least
      one integration test drives a synthetic frame through the full
      capture → backend → `on_frame` → publish path with a real (not fixture-mocked) backend
      instance, extending `tests/integration`.
- [x] Documentation: update `docs/architecture.md` §5 to reflect that CPU-only MediaPipe/ONNX
      inference is now implemented (not just designed for); update `docs/plugin-development.md`
      if `required_backends`/capability-negotiation behavior changed anything an app author
      needs to know; remove now-stale "Phase 2 will implement this" language from Phase 0
      docstrings in `backend_base.py`, `mediapipe_backend.py`, `onnx_backend.py`,
      `model_registry.py`.
- [x] Production-ready cleanup: no remaining `NotImplementedError` stubs in the inference
      backends; no leftover Phase 0/1 TODO comments referencing "Phase 2" now that it's done;
      `ruff check src/` and `mypy src/` clean on all touched modules.

**Done when:** the full inference path (backend manager → real MediaPipe/ONNX backends → model
registry → pipeline → launcher capability negotiation) is covered by unit and integration
tests, performs acceptably on a live webcam feed without freezing the UI, fails loudly and
containedly rather than crashing on any backend error, and carries no stale Phase 0/1
stub language.

---

## Phase 2 exit criteria
- [x] All milestones M2.1–M2.4 complete.
- [x] `BackendManager` constructs, shares, and warm-caches real `MediaPipeBackend`/`ONNXBackend`
      instances by name, driven entirely by configuration and each app's `required_backends`.
- [x] `MediaPipeBackend` (hand landmarks) and `ONNXBackend` run real CPU-only inference and
      return standardized output into `frame.results`, with `ModelRegistry` performing a real
      checksum-verified download-and-cache for at least one model.
- [x] The launcher greys out apps whose `required_backends` aren't satisfied instead of letting
      them crash at launch; a satisfied app launches and runs real inference end-to-end through
      `FramePipeline`.
- [x] The hand-tracking demo app (Phase 1's headline deliverable) now renders real hand
      landmarks live through the existing pipeline and widget/signal path, with zero changes to
      `AppPlugin`, `AppManifest`, or `apps/hand_tracking_demo`'s own code.
- [x] No plugin API redesign: `core/plugin_base.py` is unchanged from Phase 1; no gesture
      recognition or game logic was added to any app (reserved for Phase 3).
- [x] v1 remains CPU-only — no `onnxruntime-gpu`, no CUDA/DirectML, no GPU-specific code path;
      `DeviceConfig`/execution-provider lists are unchanged in shape from Phase 0, only
      exercised for real.
- [x] `ruff`, `mypy`, and `pytest` (including new backend/integration tests) all green in CI on
      Windows.
- [x] `core/` and `vision/` still contain zero Qt imports; nothing below `ui/`/`apps/` imports
      upward; `apps/` folders remain flat with no cross-app imports.
- [ ] Tagged as a Phase 2 checkpoint before Phase 3 begins.
