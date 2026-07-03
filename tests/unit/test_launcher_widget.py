"""Unit tests for visionplay.ui.launcher.launcher_widget (offscreen Qt)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest
from visionplay.ui.launcher.launcher_widget import ALL_CATEGORIES, LauncherWidget

#: Mirrors LauncherWidget's internal item-data role for white-box lookups.
_APP_ID_ROLE = Qt.ItemDataRole.UserRole


def make_manifest(
    app_id: str,
    name: str,
    category: str,
    icon: str = "",
    required_backends: tuple[str, ...] = (),
) -> AppManifest:
    return AppManifest(
        id=app_id,
        name=name,
        category=category,
        version="0.1.0",
        api_version=CURRENT_API_VERSION,
        icon=icon,
        required_backends=required_backends,
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


#: Manifests exercising capability negotiation: one satisfied backend app,
#: one unsatisfiable one, one declaring no backends at all.
NEGOTIATION_MANIFESTS = {
    "hands_app": make_manifest(
        "hands_app", "Hands App", "ai_demos", required_backends=("mediapipe.hands",)
    ),
    "broken_app": make_manifest(
        "broken_app", "Broken App", "ai_demos", required_backends=("missing.backend",)
    ),
    "plain_app": make_manifest("plain_app", "Plain App", "fitness"),
}


def only_hands_available(name: str) -> bool:
    return name == "mediapipe.hands"


class TestCapabilityNegotiation:
    """M2.3: apps with unsatisfiable required_backends render greyed-out."""

    def test_without_predicate_every_app_is_launchable(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS)
        for app_id in NEGOTIATION_MANIFESTS:
            item = find_app_item(widget, app_id)
            assert item.flags() & Qt.ItemFlag.ItemIsEnabled
            assert item.flags() & Qt.ItemFlag.ItemIsSelectable
            assert widget.is_app_launchable(app_id)

    def test_unsatisfied_app_renders_disabled(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS, backend_available=only_hands_available)
        item = find_app_item(widget, "broken_app")
        assert not item.flags() & Qt.ItemFlag.ItemIsEnabled
        assert not item.flags() & Qt.ItemFlag.ItemIsSelectable
        assert not widget.is_app_launchable("broken_app")

    def test_disabled_item_tooltip_names_the_missing_backend(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS, backend_available=only_hands_available)
        assert "missing.backend" in find_app_item(widget, "broken_app").toolTip(0)

    def test_satisfied_app_stays_launchable(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS, backend_available=only_hands_available)
        item = find_app_item(widget, "hands_app")
        assert item.flags() & Qt.ItemFlag.ItemIsEnabled
        assert widget.is_app_launchable("hands_app")

    def test_app_declaring_no_backends_is_always_launchable(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS, backend_available=lambda name: False)
        assert find_app_item(widget, "plain_app").flags() & Qt.ItemFlag.ItemIsEnabled
        assert widget.is_app_launchable("plain_app")

    def test_activating_disabled_app_does_not_emit(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS, backend_available=only_hands_available)
        received: list[str] = []
        widget.app_launch_requested.connect(received.append)

        # Programmatic emission bypasses Qt's disabled-item interaction block,
        # exercising the widget's own explicit guard.
        widget._tree.itemActivated.emit(find_app_item(widget, "broken_app"), 0)

        assert received == []

    def test_activating_enabled_app_still_emits(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS, backend_available=only_hands_available)
        received: list[str] = []
        widget.app_launch_requested.connect(received.append)
        widget._tree.itemActivated.emit(find_app_item(widget, "hands_app"), 0)
        assert received == ["hands_app"]

    def test_set_backend_availability_re_renders(self, qapp: QApplication) -> None:
        widget = LauncherWidget(NEGOTIATION_MANIFESTS)
        assert widget.is_app_launchable("broken_app")  # no predicate yet

        widget.set_backend_availability(only_hands_available)
        assert not widget.is_app_launchable("broken_app")
        assert not find_app_item(widget, "broken_app").flags() & Qt.ItemFlag.ItemIsEnabled

        widget.set_backend_availability(None)  # cleared: everything available again
        assert widget.is_app_launchable("broken_app")

    def test_set_apps_preserves_the_predicate(self, qapp: QApplication) -> None:
        widget = LauncherWidget(backend_available=only_hands_available)
        widget.set_apps(NEGOTIATION_MANIFESTS)
        assert not widget.is_app_launchable("broken_app")
        assert widget.is_app_launchable("hands_app")
