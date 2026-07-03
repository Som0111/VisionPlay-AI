"""Launcher/dashboard widget: lists discovered apps, signals launch intent (M1.5, M2.3).

Populated from a plain ``{app_id: AppManifest}`` mapping — typically
:attr:`~visionplay.core.plugin_registry.PluginRegistry.manifests` — rather
than a live registry reference, so this widget never imports
``PluginRegistry`` and cannot call into it. Selecting an app only emits
:attr:`LauncherWidget.app_launch_requested`; deciding what to do about it
(stopping any previously active app, starting the new one, wiring the
frame pipeline) is the application bootstrap's job (M1.6,
``docs/architecture.md`` §3 — the launcher signals intent, it does not
own app lifecycle).

Capability negotiation (M2.3): each app's ``required_backends`` are checked
against an injected availability predicate — in production
``BackendManager.is_available`` (``docs/architecture.md`` §4), injected the
same decoupled way as the manifests mapping so the widget never imports
``BackendManager`` either. An app with an unsatisfiable backend renders
disabled/greyed-out (with a tooltip naming what's missing) and never emits
:attr:`LauncherWidget.app_launch_requested`, replacing the
"every app renders as launchable" behavior explicitly deferred from M1.5.
With no predicate set, everything is treated as available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QComboBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from visionplay.core.plugin_base import AppManifest

__all__ = ["BackendAvailability", "LauncherWidget"]

#: Answers "can this backend name run right now?" — in production
#: ``BackendManager.is_available``, which never raises; any predicate
#: supplied here must honor that same no-raise contract.
BackendAvailability = Callable[[str], bool]

#: Category-filter option meaning "show every category".
ALL_CATEGORIES: str = "All Categories"

#: Item-data role used to stash each app-entry item's app id.
_APP_ID_ROLE = Qt.ItemDataRole.UserRole


class LauncherWidget(QWidget):
    """Grouped, filterable app list; emits launch intent, owns no lifecycle.

    Category header rows are unselectable and never emit
    :attr:`app_launch_requested` — only activating an actual app entry does.
    """

    #: Emitted with the app's manifest id when the user activates an app
    #: entry (double-click, or Enter/Return with it focused). Never emitted
    #: for category header rows.
    app_launch_requested = Signal(str)

    def __init__(
        self,
        manifests: Mapping[str, AppManifest] | None = None,
        parent: QWidget | None = None,
        *,
        backend_available: BackendAvailability | None = None,
    ) -> None:
        """Build the widget, optionally pre-populated with ``manifests``.

        Args:
            manifests: Apps to display initially, keyed by id.
            parent: Standard Qt parent widget.
            backend_available: Per-backend availability predicate for
                capability negotiation (production:
                ``BackendManager.is_available``). ``None`` treats every
                backend as available — all apps render launchable.
        """
        super().__init__(parent)
        self._manifests: dict[str, AppManifest] = {}
        self._backend_available = backend_available

        self._category_filter = QComboBox()
        self._category_filter.currentTextChanged.connect(self._apply_category_filter)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemActivated.connect(self._on_item_activated)

        layout = QVBoxLayout(self)
        layout.addWidget(self._category_filter)
        layout.addWidget(self._tree)

        self.set_apps(manifests or {})

    @property
    def manifests(self) -> Mapping[str, AppManifest]:
        """The apps currently displayed, keyed by id."""
        return self._manifests

    def set_apps(self, manifests: Mapping[str, AppManifest]) -> None:
        """Replace the displayed apps and rebuild the grouped tree/filter.

        Args:
            manifests: App id -> manifest — pass
                ``PluginRegistry.manifests`` in production, or any fixture
                mapping in tests.
        """
        self._manifests = dict(manifests)
        self._rebuild_category_filter()
        self._rebuild_tree()

    def set_backend_availability(self, backend_available: BackendAvailability | None) -> None:
        """Install (or clear) the capability-negotiation predicate and re-render.

        Args:
            backend_available: Per-backend availability predicate, or ``None``
                to treat every backend as available again.
        """
        self._backend_available = backend_available
        self._rebuild_tree()

    def is_app_launchable(self, app_id: str) -> bool:
        """Return ``True`` if the app's ``required_backends`` are all available.

        Args:
            app_id: Id of a displayed app.

        Raises:
            KeyError: If ``app_id`` is not among the displayed apps.
        """
        return not self._missing_backends(self._manifests[app_id])

    def _missing_backends(self, manifest: AppManifest) -> tuple[str, ...]:
        """Return the manifest's declared backends the predicate reports missing."""
        if self._backend_available is None:
            return ()
        return tuple(
            name for name in manifest.required_backends if not self._backend_available(name)
        )

    def _rebuild_category_filter(self) -> None:
        """Repopulate the category dropdown, keeping the current choice if valid."""
        categories = sorted({manifest.category for manifest in self._manifests.values()})
        previous = self._category_filter.currentText() or ALL_CATEGORIES

        self._category_filter.blockSignals(True)
        self._category_filter.clear()
        self._category_filter.addItem(ALL_CATEGORIES)
        self._category_filter.addItems(categories)
        restored_index = self._category_filter.findText(previous)
        self._category_filter.setCurrentIndex(restored_index if restored_index >= 0 else 0)
        self._category_filter.blockSignals(False)

    def _rebuild_tree(self) -> None:
        """Rebuild the tree: one unselectable header row per category, apps under it."""
        self._tree.clear()
        by_category: dict[str, list[AppManifest]] = {}
        for manifest in self._manifests.values():
            by_category.setdefault(manifest.category, []).append(manifest)

        for category in sorted(by_category):
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(category_item)
            for manifest in sorted(by_category[category], key=lambda m: m.name):
                app_item = QTreeWidgetItem([manifest.name])
                app_item.setData(0, _APP_ID_ROLE, manifest.id)
                if manifest.icon:
                    app_item.setIcon(0, QIcon(manifest.icon))
                missing = self._missing_backends(manifest)
                if missing:
                    # Capability negotiation (M2.3): unsatisfiable apps render
                    # greyed-out and unselectable instead of crashing at launch.
                    app_item.setFlags(
                        app_item.flags()
                        & ~(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    )
                    app_item.setToolTip(0, f"Unavailable — missing backends: {', '.join(missing)}")
                category_item.addChild(app_item)

        self._tree.expandAll()
        self._apply_category_filter(self._category_filter.currentText())

    def _apply_category_filter(self, category: str) -> None:
        """Show/hide top-level category rows to match the dropdown's choice."""
        show_all = category in ("", ALL_CATEGORIES)
        for index in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(index)
            item.setHidden(not show_all and item.text(0) != category)

    def _on_item_activated(self, item: QTreeWidgetItem, column: int) -> None:
        """Emit ``app_launch_requested`` for an activated app row.

        Category headers and greyed-out (capability-unsatisfied) apps never
        emit — the disabled flag already blocks normal UI activation, but the
        explicit guard also covers programmatic ``itemActivated`` emission.
        """
        app_id = item.data(0, _APP_ID_ROLE)
        if app_id is None or not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        self.app_launch_requested.emit(app_id)
