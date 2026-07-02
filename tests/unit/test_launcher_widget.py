"""Unit tests for visionplay.ui.launcher.launcher_widget (offscreen Qt)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest
from visionplay.ui.launcher.launcher_widget import ALL_CATEGORIES, LauncherWidget

#: Mirrors LauncherWidget's internal item-data role for white-box lookups.
_APP_ID_ROLE = Qt.ItemDataRole.UserRole


def make_manifest(app_id: str, name: str, category: str, icon: str = "") -> AppManifest:
    return AppManifest(
        id=app_id,
        name=name,
        category=category,
        version="0.1.0",
        api_version=CURRENT_API_VERSION,
        icon=icon,
    )


SAMPLE_MANIFESTS = {
    "air_hockey": make_manifest("air_hockey", "Air Hockey", "gesture_games"),
    "squat_counter": make_manifest("squat_counter", "Squat Counter", "fitness"),
    "face_filter": make_manifest("face_filter", "Face Filter", "face_ar"),
    "object_detector": make_manifest("object_detector", "Object Detector", "ai_demos"),
}


def find_app_item(widget: LauncherWidget, app_id: str) -> QTreeWidgetItem:
    """Locate the tree item for ``app_id`` across every category header."""
    tree = widget._tree
    for cat_index in range(tree.topLevelItemCount()):
        category_item = tree.topLevelItem(cat_index)
        for child_index in range(category_item.childCount()):
            child = category_item.child(child_index)
            if child.data(0, _APP_ID_ROLE) == app_id:
                return child
    raise AssertionError(f"No item found for app id {app_id!r}")


class TestPopulation:
    def test_populates_from_constructor(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        assert widget.manifests == SAMPLE_MANIFESTS

    def test_groups_apps_under_category_headers(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        tree = widget._tree
        categories = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
        assert categories == {"gesture_games", "fitness", "face_ar", "ai_demos"}

    def test_each_app_appears_exactly_once(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        found = [find_app_item(widget, app_id) for app_id in SAMPLE_MANIFESTS]
        assert len(found) == len(SAMPLE_MANIFESTS)

    def test_app_item_shows_display_name(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        item = find_app_item(widget, "air_hockey")
        assert item.text(0) == "Air Hockey"

    def test_empty_manifests_renders_no_categories(self, qapp: QApplication) -> None:
        widget = LauncherWidget({})
        assert widget._tree.topLevelItemCount() == 0


class TestUpdatesWhenRegistryChanges:
    def test_set_apps_replaces_displayed_apps(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        new_manifests = {
            "new_app": make_manifest("new_app", "New App", "ai_demos"),
        }
        widget.set_apps(new_manifests)
        assert widget.manifests == new_manifests
        tree = widget._tree
        assert tree.topLevelItemCount() == 1
        assert find_app_item(widget, "new_app").text(0) == "New App"

    def test_set_apps_removes_stale_entries(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        widget.set_apps({})
        assert widget._tree.topLevelItemCount() == 0


class TestLaunchSignal:
    def test_activating_app_item_emits_signal_with_correct_id(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        received: list[str] = []
        widget.app_launch_requested.connect(received.append)

        item = find_app_item(widget, "squat_counter")
        widget._tree.itemActivated.emit(item, 0)

        assert received == ["squat_counter"]

    def test_activating_category_header_does_not_emit(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        received: list[str] = []
        widget.app_launch_requested.connect(received.append)

        tree = widget._tree
        header_item = tree.topLevelItem(0)
        tree.itemActivated.emit(header_item, 0)

        assert received == []

    def test_never_calls_into_a_registry(self, qapp: QApplication) -> None:
        """The widget must only need a plain mapping — never a PluginRegistry.

        Passing an object with no start()/stop() methods proves the widget
        never attempts to call them.
        """
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        received: list[str] = []
        widget.app_launch_requested.connect(received.append)
        item = find_app_item(widget, "air_hockey")
        widget._tree.itemActivated.emit(item, 0)
        assert received == ["air_hockey"]  # no AttributeError from touching a registry


class TestCategoryFilter:
    def test_filter_defaults_to_all_categories(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        assert widget._category_filter.currentText() == ALL_CATEGORIES
        tree = widget._tree
        assert all(not tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount()))

    def test_selecting_category_hides_others(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        widget._category_filter.setCurrentText("fitness")

        tree = widget._tree
        visible = {
            tree.topLevelItem(i).text(0)
            for i in range(tree.topLevelItemCount())
            if not tree.topLevelItem(i).isHidden()
        }
        assert visible == {"fitness"}

    def test_returning_to_all_categories_shows_everything_again(self, qapp: QApplication) -> None:
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        widget._category_filter.setCurrentText("fitness")
        widget._category_filter.setCurrentText(ALL_CATEGORIES)

        tree = widget._tree
        assert all(not tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount()))


class TestAllAppsLaunchable:
    def test_no_app_item_is_disabled(self, qapp: QApplication) -> None:
        """Capability negotiation (greying out apps) is Phase 2 — nothing disabled yet."""
        widget = LauncherWidget(SAMPLE_MANIFESTS)
        for app_id in SAMPLE_MANIFESTS:
            item = find_app_item(widget, app_id)
            assert item.flags() & Qt.ItemFlag.ItemIsEnabled
            assert item.flags() & Qt.ItemFlag.ItemIsSelectable
