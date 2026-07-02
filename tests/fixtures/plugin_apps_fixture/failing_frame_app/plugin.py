"""Fixture plugin whose on_frame() always raises."""

from tests.fixtures.plugin_apps_fixture._support import RecordingPlugin


class Plugin(RecordingPlugin):
    fail_on_frame = True
