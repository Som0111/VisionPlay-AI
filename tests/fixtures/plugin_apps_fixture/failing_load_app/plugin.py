"""Fixture plugin whose on_load() always raises."""

from tests.fixtures.plugin_apps_fixture._support import RecordingPlugin


class Plugin(RecordingPlugin):
    fail_on_load = True
