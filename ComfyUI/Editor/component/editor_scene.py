import sys
import math
from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtGui import QBrush, QColor, QPen, QPainter
from PySide6.QtCore import QLine

class editorScene(QGraphicsScene):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.view = None
        # 

    def drawBackground(self, painter: QPainter, rect) -> None:
        self.setBackgroundBrush(QBrush(QColor('#212121')))
        self.width = 32000
        self.height = 32000
        self.grid_size = 20
        self.chunk_size = 10
        self.setSceneRect(-self.width / 2, -self.height / 2, self.width, self.height)
        self.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        self.normal_line_pen = QPen(QColor("#313131"))
        self.normal_line_pen.setWidthF(1.0)
        self.dark_line_pen = QPen(QColor("#151515"))
        self.dark_line_pen.setWidthF(1.5)

        super().drawBackground(painter, rect)
        lines, drak_lines = self.cal_grid_lines(rect)
        painter.setPen(self.normal_line_pen)
        painter.drawLines(lines)

        painter.setPen(self.dark_line_pen)
        painter.drawLines(drak_lines)

    def cal_grid_lines(self, rect):
        left, right, top, bottom = math.floor(rect.left()), math.floor(
            rect.right()), math.floor(rect.top()), math.floor(rect.bottom())

        first_left = left - (left % self.grid_size)
        first_top = top - (top % self.grid_size)

        lines = []
        drak_lines = []
        for v in range(first_top, bottom, self.grid_size):
            line = QLine(left, v, right, v)
            if v % (self.grid_size * self.chunk_size) == 0:
                drak_lines.append(line)
            else:
                lines.append(line)

        for h in range(first_left, right, self.grid_size):
            line = QLine(h, top, h, bottom)
            if h % (self.grid_size * self.chunk_size) == 0:
                drak_lines.append(line)
            else:
                lines.append(line)

        return lines, drak_lines