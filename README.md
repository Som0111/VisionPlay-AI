# VisionPlay AI

A production-quality desktop platform for computer vision games, utilities, and AI demos —
built with Python, OpenCV, MediaPipe, ONNX Runtime, and PySide6.

VisionPlay AI hosts many small, self-contained CV apps (gesture-controlled games,
fitness/pose utilities, face/AR filters, and general AI vision demos) behind a single
launcher/dashboard. Each app is a plugin implementing a shared interface, built on top of a
common camera pipeline and inference backend layer.

Windows-only for v1; the core is designed to be portable to macOS/Linux later without a rewrite.

## Status

Early architecture/scaffolding stage — see `docs/roadmap.md` for the phase plan. No runnable
app yet.

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — project conventions and guidance for AI-assisted development
- [`docs/architecture.md`](./docs/architecture.md) — full architecture, folder structure, risks
- [`docs/plugin-development.md`](./docs/plugin-development.md) — how to build a new app/plugin
- [`docs/roadmap.md`](./docs/roadmap.md) — phased development plan

## Privacy

Camera frames are processed entirely on-device and never leave the machine. Any telemetry is
opt-in and metadata-only.

## License

[MIT](./LICENSE)
