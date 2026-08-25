"""Accessibility helpers for the Qt download manager."""

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent


class _KeyboardActivationFilter(QObject):
	def __init__(self, activate: Callable[[], None], parent=None, allow_alt_down: bool = False):
		super().__init__(parent)
		self.activate = activate
		self.allow_alt_down = allow_alt_down

	def eventFilter(self, watched, event):  # noqa: N802 - Qt virtual method name
		if event.type() == QEvent.Type.KeyPress:
			key = event.key()
			is_alt_down = self.allow_alt_down and key == Qt.Key.Key_Down and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
			if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) or is_alt_down:
				if not event.isAutoRepeat():
					self.activate()
				return True

		return super().eventFilter(watched, event)


def set_accessible(widget, name: str, description: str = "", tooltip: str | None = None, focusable: bool = False):
	"""Set the properties used by Qt Accessibility and keyboard users."""
	widget.setAccessibleName(name)
	widget.setAccessibleDescription(description)
	if tooltip is not None:
		widget.setToolTip(tooltip)
	if focusable:
		widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
	return widget


def make_keyboard_activatable(widget, activate: Callable[[], None], *, allow_alt_down: bool = False):
	"""Give a custom QWidget Enter/Space activation behavior."""
	set_accessible(widget, widget.accessibleName(), widget.accessibleDescription(), focusable=True)
	widget.installEventFilter(_KeyboardActivationFilter(activate, widget, allow_alt_down))
	return widget


def add_accessible_action(command_bar, action, description: str = ""):
	"""Add a command action and name the custom-painted tool button it creates."""
	name = action.text()
	action.setToolTip(name)
	action.setStatusTip(description)
	button = command_bar.addAction(action)
	set_accessible(button, name, description, tooltip=name, focusable=True)
	return button


def configure_command_bar(command_bar, more_actions_name: str):
	set_accessible(command_bar, command_bar.accessibleName(), command_bar.accessibleDescription())
	set_accessible(command_bar.moreButton, more_actions_name, tooltip=more_actions_name, focusable=True)


def announce(widget, message: str, assertive: bool = False):
	"""Send a live announcement to the active platform screen reader."""
	if not message:
		return

	event = QAccessibleAnnouncementEvent(widget, message)
	politeness = QAccessible.AnnouncementPoliteness.Assertive if assertive else QAccessible.AnnouncementPoliteness.Polite
	event.setPoliteness(politeness)
	QAccessible.updateAccessibility(event)


def accessible_info_bar(kind: str, *, title: str, content: str, parent, close_text: str = "Close", **kwargs):
	"""Show an InfoBar and expose its message as an accessibility announcement."""
	from qfluentwidgets import InfoBar

	bar = getattr(InfoBar, kind)(title=title, content=content, parent=parent, **kwargs)
	message = ": ".join(part for part in (title, content) if part)
	set_accessible(bar, message or content)
	if hasattr(bar, "closeButton"):
		set_accessible(bar.closeButton, close_text, tooltip=close_text, focusable=True)
	announce(parent or bar, message or content, assertive=kind in {"error", "warning"})
	return bar
