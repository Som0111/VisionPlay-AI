# VisionPlay AI — Development Roadmap

| Phase | Goal |
|---|---|
| 0 — Foundation | Repo scaffold, CI skeleton, core (config/logging/paths/event bus), camera-to-Qt render pipeline proven end-to-end, **PyInstaller packaging spike**. Session-by-session breakdown: [`phase-0-checklist.md`](./checklists/phase-0-checklist.md) |
| 1 — Plugin skeleton | `AppPlugin`/`AppManifest`/registry, launcher UI, `_template` app, one trivial hand-tracking "hello world" through the full pipeline. Session-by-session breakdown: [`phase-1-checklist.md`](./checklists/phase-1-checklist.md) |
| 2 — Inference backends | `InferenceBackend` ABC (with a `device` param even though v1 only implements `cpu`), CPU-only MediaPipe + ONNX implementations, model registry, capability negotiation in launcher. Session-by-session breakdown: [`phase-2-checklist.md`](./checklists/phase-2-checklist.md) |
| 3 — Vertical slice | Exactly **one real app per category** (gesture game, fitness counter, face filter, AI demo) to prove the abstractions generalize before parallelizing further; check CPU-only FPS is acceptable for the AI-demo app in particular |
| 4 — Polish | Theming, global settings dialog (device selector shows only "CPU" for v1 — the field exists, just with one option), crash reporter, structured logging |
| 5 — Release | Windows build validated on a clean VM, docs/screenshots, tag v0.1.0 |
| 6 — Stretch (backlog) | GPU acceleration (DirectML/CUDA execution providers, MediaPipe GPU delegate — additive per the §5 design, not a rewrite), external plugin loader, macOS/Linux packaging, auto-update |

## Why Phase 3 is narrow

Phase 3 is deliberately one app per category, not a full catalog, so the plugin/backend
abstractions get validated across genuinely different workloads (game loop vs. rep-counting
vs. AR overlay vs. detection) before more apps are built on top of a design that hasn't been
stress-tested yet.

## Candidate apps per category (post-v1 backlog, pick from as capacity allows)

- **Gesture-controlled games** — air hockey, hand-tracked fruit-slice, air drawing, gesture puzzles
- **Fitness/pose utilities** — squat/pushup rep counter, posture checker, pose-based form feedback
- **Face/AR filters** — face-mesh filters, virtual try-on, background replacement
- **AI vision demos/utilities** — object detection, OCR, hand-sign recognition, general showcase demos

## Resolved decisions

- **License**: MIT ([LICENSE](../LICENSE))
- **Inference backend for v1**: CPU-first. See `docs/architecture.md` § 5 for how the backend
  abstraction keeps GPU acceleration a later, additive change rather than a redesign.

## Open questions to resolve before/during implementation

- Any specific "flagship" app to prioritize for a demo reel/portfolio highlight
