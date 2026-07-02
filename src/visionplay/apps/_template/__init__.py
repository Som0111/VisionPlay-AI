"""Scaffold for new apps — copy this whole folder to `apps/<app_name>/`.

Underscore-prefixed by design: `PluginRegistry.discover` (M1.2) skips any
app package whose name starts with ``_`` (``core/plugin_registry.py``),
exactly like the real ``_template`` this mirrors in
``docs/architecture.md`` §2 and §3. This package is never registered and
never appears in the launcher.

See ``docs/plugin-development.md`` § "Adding a new app — checklist" for
the full copy/fill-in/test steps.
"""
