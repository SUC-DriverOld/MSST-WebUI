from PySide6.QtWidgets import QGraphicsItem
from PySide6.QtGui import QColor, QPainter, QBrush, QPen, QFont
from PySide6.QtCore import QRectF, Qt
import os
import shutil
import sys
sys.path.append("D:\projects\python\MSST-WebUI")
from ComfyUI.Editor.common.config import color, font


class FileDragArea(QGraphicsItem):
    def __init__(self, parent=None, path="./tmp/input"):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.folder_list = []
        self.file_list = []
        self.width = 200
        self.height = 150
        self.path = path
        self.setToolTip(f"folder list: {self.folder_list}\nfile list: {self.file_list}")

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter, option, widget):
        # 设置整体边框的外矩形
        margin = 5  # 边框与内容的距离
        outer_rect = self.boundingRect().adjusted(margin, margin, -margin, -margin)

        # 整体背景和边框
        painter.setBrush(Qt.NoBrush)  # 背景透明
        pen = QPen(color)  # 边框颜色
        pen.setStyle(Qt.DashLine)  # 设置虚线边框
        pen.setWidth(2)  # 边框宽度
        painter.setPen(pen)
        painter.drawRoundedRect(outer_rect, 10, 10)  # 绘制外层整体圆角矩形

        # 主区域（上部分）
        main_rect = QRectF(
            outer_rect.x(),
            outer_rect.y(),
            outer_rect.width(),
            outer_rect.height() * 2 / 3
        )
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(main_rect, Qt.AlignCenter, "drop files or \nfolders here")

        # 清空区域（下部分）
        clear_rect = QRectF(
            outer_rect.x(),
            outer_rect.y() + outer_rect.height() * 2 / 3,
            outer_rect.width(),
            outer_rect.height() / 3
        )
        painter.setFont(font)
        painter.drawText(clear_rect, Qt.AlignCenter, "clear")

        pen = QPen(color)
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(15, self.height * 2 / 3, self.width - 15, self.height * 2 / 3)

    def mousePressEvent(self, event):
        # 判断点击位置是否在清空区域内
        margin = 5
        outer_rect = self.boundingRect().adjusted(margin, margin, -margin, -margin)
        clear_rect = QRectF(
            outer_rect.x(),
            outer_rect.y() + outer_rect.height() * 2 / 3,
            outer_rect.width(),
            outer_rect.height() / 3
        )
        if clear_rect.contains(event.pos()):  # 点击在清空区域
            self.file_list.clear()
            self.folder_list.clear()
            print("clear all files and folders.")
            # 更新界面
            self.update()
            self.updateToolTip()

    def updateToolTip(self):
        self.setToolTip(f"folder list: {self.folder_list}\nfile list: {self.file_list}")        

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    self.folder_list.append(path)
                elif os.path.isfile(path):
                    self.file_list.append(path)
            print("Updated file list:", self.file_list)
            print("Updated folder list:", self.folder_list)
            self.updateToolTip()
            event.acceptProposedAction()
    
    def copyFiles(self):
        """Clear the target path and copy all files and folders."""
        # 清空目标文件夹
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)

        # 复制文件
        for file in self.file_list:
            dest_path = os.path.join(self.path, os.path.basename(file))
            shutil.copy(file, dest_path)

        # 复制文件夹
        for folder in self.folder_list:
            folder_name = os.path.basename(folder)
            dest_folder = os.path.join(self.path, folder_name)
            shutil.copytree(folder, dest_folder)

        print(f"All files and folders copied to {self.path}")



if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene

    app = QApplication(sys.argv)
    view = QGraphicsView()
    scene = QGraphicsScene()
    view.setScene(scene)
    view.setFixedSize(800, 600)
    view.setRenderHint(QPainter.Antialiasing)
    view.show()
    drag_area = FileDragArea()
    scene.addItem(drag_area)
    sys.exit(app.exec())        