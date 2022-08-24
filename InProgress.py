from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QLabel, QLineEdit, QProgressBar, \
    QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QDialogButtonBox
from PyQt5.QtCore import Qt, QBasicTimer, QThread, QRect
import sys


class ProgressBar(QDialog):
    def __init__(self, parent=None):
        super(ProgressBar, self).__init__(parent)


        self.resize(500, 32)
        self.setWindowModality(Qt.ApplicationModal)


        self.progressBar = QProgressBar(self)

        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.setWindowTitle(self.tr('Image in progress'))
        self.progressBar.setGeometry(QRect(1, 3, 499, 28))
        self.show()

    def set_value(self,value):

        self.progressBar.setValue(value)
        QApplication.processEvents()