import sys
import math

sys.path.append('.')

from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtGui import QBrush, QColor, QPen, QPainter
from PySide6.QtCore import Qt, QLine
from config import EditorConfig


class ComfyGUIScene(QGraphicsScene):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.setBackgroundBrush(QBrush(QColor('#212121')))

        self._width = EditorConfig.editor_scene_width
        self._height = EditorConfig.editor_scene_height
        self._grid_size = EditorConfig.editor_scene_grid_size
        self._chunk_size = EditorConfig.editor_scene_grid_chunk
        self.setSceneRect(-self._width / 2, -self._height / 2, self._width, self._height)

        # 网格背景
        self._normal_line_pen = QPen(QColor(EditorConfig.editor_scene_grid_normal_line_color))
        self._normal_line_pen.setWidthF(EditorConfig.editor_scene_grid_normal_line_width)
        self._dark_line_pen = QPen(QColor(EditorConfig.editor_scene_grid_dark_line_color))
        self._dark_line_pen.setWidthF(EditorConfig.editor_scene_grid_dark_line_width)

        self.setItemIndexMethod(QGraphicsScene.NoIndex)

    def set_view(self, view):
        self._view = view

    def get_view(self):
        return self._view

    def drawBackground(self, painter: QPainter, rect) -> None:
        super().drawBackground(painter, rect)

        lines, drak_lines = self.cal_grid_lines(rect)

        # 画普通的线
        painter.setPen(self._normal_line_pen)
        painter.drawLines(lines)

        # 画粗线
        painter.setPen(self._dark_line_pen)
        painter.drawLines(drak_lines)

    def cal_grid_lines(self, rect):
        left, right, top, bottom = math.floor(rect.left()), math.floor(
            rect.right()), math.floor(rect.top()), math.floor(rect.bottom())

        first_left = left - (left % self._grid_size)
        first_top = top - (top % self._grid_size)

        lines = []
        drak_lines = []
        # 画横线
        for v in range(first_top, bottom, self._grid_size):
            line = QLine(left, v, right, v)
            if v % (self._grid_size * self._chunk_size) == 0:
                drak_lines.append(line)
            else:
                lines.append(line)

        # 画竖线
        for h in range(first_left, right, self._grid_size):
            line = QLine(h, top, h, bottom)
            if h % (self._grid_size * self._chunk_size) == 0:
                drak_lines.append(line)
            else:
                lines.append(line)

        return lines, drak_lines
