from PySide6.QtWidgets import QGraphicsView, QMenu
from PySide6.QtGui import QPainter, QMouseEvent, QKeyEvent, QDragEnterEvent, QDragMoveEvent, QFont
from PySide6.QtCore import Qt, QEvent, QPointF
from edge import NodeEdge, DraggingEdge
from node import Node
from node_port import NodePort
from nodes.model_node import MSSTModelNode, VRModelNode
from nodes.data_flow_node import InputNode, OutputNode
import json


class ComfyGUIView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self._scene = scene
        self.edges = []
        self.nodes = []
        self.setScene(self._scene)
        self._scene.set_view(self)

        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.TextAntialiasing |
                            QPainter.SmoothPixmapTransform |
                            QPainter.LosslessImageRendering)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # Hide scrollbars
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # scale
        self._zoom_clamp = [0.2, 2]
        self._zoom_factor = 1.05
        self._view_scale = 1.0
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # Scale relative to mouse position
        self.setDragMode(QGraphicsView.RubberBandDrag)

        # Disable drag mode
        self._drag_mode = False
        # Draggable edges
        self._drag_edge = None
        self._drag_edge_mode = False

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat('application/json'):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasFormat('application/json'):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat('application/json'):
            item_data = event.mimeData().data('application/json')
            model_info = json.loads(item_data.data().decode('utf-8'))
            pos = self.mapToScene(event.pos()).toPoint()
            self.create_node(model_info, [pos.x(), pos.y()])
            event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self.middle_button_pressed(event)
        elif event.button() == Qt.LeftButton:
            self.left_button_pressed(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self.middle_button_released(event)
        elif event.button() == Qt.LeftButton:
            self.left_button_released(event)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self.reset_scale()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom_factor = self._zoom_factor
        else:
            zoom_factor = 1 / self._zoom_factor
        self._view_scale *= zoom_factor
        if self._view_scale < self._zoom_clamp[0] or self._view_scale > self._zoom_clamp[1]:
            zoom_factor = 1.0
            self._view_scale = self._last_scale
        self._last_scale = self._view_scale
        self.scale(zoom_factor, zoom_factor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_edge_mode:
            self._drag_edge.update_position(self.mapToScene(event.pos()))
        else:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Delete:
            self.remove_selected_items()
        else:
            return super().keyPressEvent(event)

    def middle_button_pressed(self, event):
        if self.itemAt(event.pos()) is not None:
            return
        else:
            release_event = QMouseEvent(QEvent.MouseButtonRelease, event.pos(), Qt.LeftButton, Qt.NoButton, event.modifiers())
            super().mouseReleaseEvent(release_event)

            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._drag_mode = True
            click_event = QMouseEvent(QEvent.MouseButtonPress, event.pos(), Qt.LeftButton, Qt.NoButton, event.modifiers())
            super().mousePressEvent(click_event)

    def middle_button_released(self, event):
        release_event = QMouseEvent(QEvent.MouseButtonRelease, event.pos(), Qt.LeftButton, Qt.NoButton, event.modifiers())
        super().mouseReleaseEvent(release_event)

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._drag_mode = False

    def left_button_pressed(self, event: QMouseEvent):
        mouse_pos = event.pos()
        item = self.itemAt(mouse_pos)
        if isinstance(item, NodePort):
            self._drag_edge_mode = True
            self.create_dragging_edge(item)
        else:
            super().mousePressEvent(event)

    def create_dragging_edge(self, port: NodePort):
        port_pos = port.get_port_pos()
        if port.port_type == NodePort.PORT_TYPE_INPUT or port.port_type == NodePort.PORT_TYPE_OUTPUT:
            drag_from_source = True
        else:
            drag_from_source = False
        if self._drag_edge is None:
            self._drag_edge = DraggingEdge(port_pos,
                                           port_pos,
                                           edge_color=port._port_color,
                                           drag_from_source=drag_from_source,
                                           scene=self._scene)

            self._drag_edge.set_first_port(port)
            self._scene.addItem(self._drag_edge)

    def left_button_released(self, event: QMouseEvent):
        if self._drag_edge_mode:
            self._drag_edge_mode = False
            item = self.itemAt(event.pos())
            if isinstance(item, NodePort):
                self._drag_edge.set_second_port(item)
                edge = self._drag_edge.create_node_edge()
                if edge is not None:
                    self.edges.append(edge)
            self._scene.removeItem(self._drag_edge)
            self._drag_edge = None

        super().mouseReleaseEvent(event)

    def right_button_pressed(self, event):
        pass

    def right_button_released(self, event):
        pass

    def set_menu_widget(self, widget):
        self._menu_widget = widget

    def reset_scale(self):
        self.resetTransform()
        self._view_scale = 1.0

    def add_node(self, node, pos = [0, 0], index = None):
        self._scene.addItem(node)
        node.setPos(pos[0], pos[1])
        node.set_scene(self._scene)
        if index is not None:
            node.index = index
        else:
            node.index = len(self.nodes)
        self.nodes.append(node)

    def create_node(self, model_info, pos):

        if isinstance(model_info, str):
            if model_info == "InputNode":
                new_node = InputNode()
            elif model_info == "OutputNode":
                new_node = OutputNode()
        else:
            model_class = model_info.get("model_class")
            if model_class == "vr_models":
                model_name = model_info.get("name")
                new_node = VRModelNode(model_class = model_class, model_name = model_name)
            else:
                model_class = model_info.get("model_class")
                model_type = model_info.get("model_type")
                model_name = model_info.get("name")
                new_node = MSSTModelNode(model_class = model_class, model_name = model_name, model_type = model_type)

        self.add_node(new_node, pos = pos, index = None)

    def remove_selected_items(self):
        selected_items = self._scene.selectedItems()
        for item in selected_items:
            if isinstance(item, NodeEdge):
                self.edges.remove(item)
                item.update()
            elif isinstance(item, Node):
                item.remove_edge()
                self.nodes.remove(item)
                item.update()
            self._scene.removeItem(item)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        context_menu.setFont(QFont("Consolas", 10))
        context_menu.addAction("删除选定项", self.remove_selected_items)

        context_menu.exec(event.globalPos())