"""Fixture plugin for the underscore-prefixed app; must never be constructed."""

from tests.fixtures.plugin_apps_fixture._support import RecordingPlugin


class Plugin(RecordingPlugin):
    pass
