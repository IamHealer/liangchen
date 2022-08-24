# _*_coding: UTF-8_*_
# 开发作者 ：TXH
# 开发时间 ：2020-09-08 10:20
# 文件名称 ：Qt_Processbar.py
# 开发工具 ：Python 3.7 + Pycharm IDE

from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QLabel, QLineEdit, QProgressBar, \
    QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QDialogButtonBox
from PyQt5.QtCore import Qt, QBasicTimer, QThread, QRect
import sys


class ProgressBar(QDialog):
    def __init__(self, parent=None):
        super(ProgressBar, self).__init__(parent)

        # Qdialog窗体的设置
        self.resize(500, 32)  # QDialog窗的大小
        self.setWindowModality(Qt.ApplicationModal)

        # 创建并设置 QProcessbar
        self.progressBar = QProgressBar(self)  # 创建

        self.progressBar.setMinimum(0)  # 设置进度条最小值
        self.progressBar.setMaximum(100)  # 设置进度条最大值
        self.progressBar.setValue(0)  # 进度条初始值为0
        self.setWindowTitle(self.tr('Image in progress'))
        self.progressBar.setGeometry(QRect(1, 3, 499, 28))  # 设置进度条在 QDialog 中的位置 [左，上，右，下]
        self.show()

    def set_value(self,value):  # 设置总任务进度和子任务进度

        self.progressBar.setValue(value)
        QApplication.processEvents()