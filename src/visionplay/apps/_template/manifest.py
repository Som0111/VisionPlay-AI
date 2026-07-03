"""Template manifest — copy and fill in every field for a real app.

This ``MANIFEST`` is illustrative only. It is never registered: the
``_template`` package name's underscore prefix makes
``PluginRegistry.discover`` skip it unconditionally (M1.2), regardless of
what this file contains. ``id`` is deliberately not a plausible real app
id, as a second, belt-and-suspenders signal that this manifest must not be
copied as-is.
"""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

#: TEMPLATE ONLY — never registered. Replace every field when copying this
#: file to `apps/<app_name>/manifest.py`.
MANIFEST = AppManifest(
    id="_template_do_not_register",  # TODO: unique, stable id — never reused across apps
    name="Template App",  # TODO: display name shown in the launcher
    category="ai_demos",  # TODO: one of APP_CATEGORIES (see plugin_base.py)
    version="0.1.0",  # TODO: this app's own version, independent of the platform version
    api_version=CURRENT_API_VERSION,  # keep in sync with the AppPlugin contract you target
    required_backends=(),  # TODO: e.g. ("mediapipe.hands",) — unavailable names grey the app out
    icon="assets/icon.png",  # TODO: path relative to this app's own assets/ folder
)
