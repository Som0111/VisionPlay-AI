"""Unit tests for the apps/_template scaffold (M1.3).

Covers: the manifest is well-formed but never registrable, the processor
is pure/headless-testable, the plugin delegates every lifecycle stage to
it, the widget only updates via its slot, and — the M1.3 regression this
milestone specifically calls for — the real registry never discovers this
package.
"""

import numpy as np

from visionplay.apps._template.manifest import MANIFEST
from visionplay.apps._template.plugin import Plugin
from visionplay.apps._template.processor import TemplateProcessor
from visionplay.apps._template.widget import TemplateWidget
from visionplay.core.event_bus import EventBus
from visionplay.core.plugin_base import AppPlugin
from visionplay.core.plugin_registry import PluginRegistry
from visionplay.vision.pipeline.frame_types import Frame


def make_frame(frame_id: int = 0) -> Frame:
    return Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class TestManifest:
    def test_manifest_is_well_formed(self) -> None:
        assert MANIFEST.id == "_template_do_not_register"
        assert MANIFEST.category == "ai_demos"

    def test_manifest_id_is_not_a_plausible_real_app_id(self) -> None:
        # Belt-and-suspenders: even if someone copies this file without
        # renaming the id, it reads unmistakably as "do not use this".
        assert "do_not_register" in MANIFEST.id


class TestProcessor:
    def test_process_returns_frame_unchanged(self) -> None:
        processor = TemplateProcessor()
        frame = make_frame(3)
        assert processor.process(frame) is frame

    def test_start_and_stop_do_not_raise(self) -> None:
        processor = TemplateProcessor()
        processor.start()
        processor.stop()


class TestPlugin:
    def test_is_an_app_plugin(self) -> None:
        assert isinstance(Plugin(), AppPlugin)

    def test_on_start_delegates_to_processor(self) -> None:
        plugin = Plugin()
        plugin.on_start()  # must not raise; delegates to TemplateProcessor.start

    def test_on_frame_delegates_to_processor_and_returns_frame(self) -> None:
        plugin = Plugin()
        frame = make_frame(7)
        assert plugin.on_frame(frame) is frame

    def test_full_lifecycle_does_not_raise(self) -> None:
        plugin = Plugin()
        plugin.on_load()
        plugin.on_start()
        plugin.on_frame(make_frame())
        plugin.on_stop()
        plugin.on_unload()


class TestWidget:
    def test_on_frame_ready_updates_label_text(self, qapp: object) -> None:
        widget = TemplateWidget()
        widget.on_frame_ready(make_frame(42))
        assert "42" in widget._status_label.text()


class TestDiscoveryExclusion:
    def test_real_registry_never_discovers_template(self) -> None:
        """Regression for M1.3: the underscore-prefixed folder name alone
        must keep `_template` out of the real apps/ discovery, independent
        of the fixture-based coverage in test_plugin_registry.py."""
        registry = PluginRegistry(event_bus=EventBus())  # default apps_package="visionplay.apps"
        registry.discover()
        assert "_template_do_not_register" not in registry.manifests
        assert all(not app_id.startswith("_template") for app_id in registry.manifests)
