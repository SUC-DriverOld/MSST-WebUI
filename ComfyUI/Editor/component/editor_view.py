from ComfyUI.Editor.component.editor_scene import EditorScene
from ComfyUI.Editor.common.config import cfg
from PySide6.QtWidgets import QGraphicsView, QSizePolicy
from PySide6.QtGui import QPainter, QMouseEvent
from PySide6.QtCore import Qt
from ComfyUI.Editor.component.node_port import InputPort, OutputPort

class EditorView(QGraphicsView):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.scene = EditorScene(self)
        self.setScene(self.scene)
        self.setupParameters()
        self.setupPolicy()
        # self.setStyleSheet("""
        #     border: 2px solid #39c5bb;
        #     border-radius: 20px;
        # """)
        self._last_mouse_pos = None
        self.setAcceptDrops(True)

    def setupParameters(self):
        self._zoom_clamp = [0.2, 2]
        self._view_scale = 1.0
        self._pan_sensitivity = cfg.get(cfg.pan_sensitivity)
        print(self._pan_sensitivity)

    def setupPolicy(self):
        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.TextAntialiasing |
                            QPainter.SmoothPixmapTransform |
                            QPainter.LosslessImageRendering)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)

    # 滚轮缩放相关

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_scale = self._view_scale * zoom_factor
        if self._zoom_clamp[0] <= new_scale <= self._zoom_clamp[1]:
            self._view_scale = new_scale
            self.scale(zoom_factor, zoom_factor)

    def resetScale(self):
        reset_factor = 1.0 / self._view_scale
        self.scale(reset_factor, reset_factor)
        self._view_scale = 1.0        
    
    # 鼠标事件相关

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)  # 切换到平移模式
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)  # 恢复默认的选择模式
            self._last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse_pos is not None:
            delta = event.pos() - self._last_mouse_pos
            delta_scaled = delta * self._pan_sensitivity
            self.translate(-delta_scaled.x(), -delta_scaled.y())
            self._last_mouse_pos = event.pos()
        else:
            super().mouseMoveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
            text = event.mimeData().text()
            pos = self.mapToScene(event.pos())
            self.scene.addNodeFromText(text, pos)
        else:
            event.ignore()

