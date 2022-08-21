from PyQt5.QtWidgets import QApplication
import sys
from UISetting import Stats

if __name__ == "__main__":

    app = QApplication(sys.argv)
    stats = Stats()
    stats.show()
    app.exec_()