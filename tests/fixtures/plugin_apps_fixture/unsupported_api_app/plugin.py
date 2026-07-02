"""Fixture plugin for the unsupported-api_version app.

Never expected to be constructed — the registry must reject this app
based on ``manifest.py`` alone, before importing this module's ``Plugin``.
"""

from tests.fixtures.plugin_apps_fixture._support import RecordingPlugin


class Plugin(RecordingPlugin):
    pass
