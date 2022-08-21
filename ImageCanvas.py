from PyQt5 import QtCore
from PyQt5.QtGui import *


import numpy as np

from PyQt5.Qt import QPainter, QPoint, QPen
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.Qt import QWidget, QColor, QPixmap,  QSize
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel, QMessageBox, QHBoxLayout
import cv2

class PaintBoard(QLabel):

    def __init__(self, Parent=None,image=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        print('using myLabel')
        self.__InitView()
        self.__InitData(image)

    def __InitView(self):

        self.setMinimumSize(QSize(500, 500))
        self.setMaximumSize(QSize(800, 800))
        self.setAlignment(Qt.AlignCenter)

        # self.setStyleSheet("background-color: yellow")
        # self.setAlignment(Qt.AlignCenter)

    def __InitData(self,image):
        #
        self.filename = None
        self.W, self.H = image.shape[1],image.shape[0]
        self.drawTool = 'brush'
        self.draw = False
        self.crop = False
        self.imageHistory = []
        self.redoHistory = []
        #

        self.oriImage = image.copy()
        self.image = image.copy()
        self.tmpImage = image.copy()
        self.setPhoto(self.image)
        self.shape = self.image.shape


        self.__painter = QPainter()
        self.__thickness = 5
        self.__penColor = QColor("black")
        self.__penStyle = Qt.SolidLine

        self.__colorList = QColor.colorNames()
        self.temp = QPixmap()
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.scale =1
        width = image.shape[1]
        height = image.shape[0]
        self.ratio = height/width

    def setFileName(self,name):
        self.filename = name

    def setCrop(self,crop):
        self.crop =crop
        self.setMouse()

    def setDraw(self,draw):
        self.draw = draw
        self.setMouse()

    def setMouse(self):
        if self.crop or self.draw:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    def addHistory(self,image):
        self.imageHistory.append(image)

    def emptyRedo(self):
        self.redoHistory = []

    def redo(self):
        if len(self.redoHistory)>0:
            img =self.redoHistory.pop()
            self.imageHistory.append(img)
            self.image = img
            self.setPhoto(img)

    def undo(self):
        # first image is ori image
        if len(self.imageHistory) > 1:
            self.redoHistory.append(self.imageHistory.pop())
            # get last operation image
            self.image = self.imageHistory[-1]
            self.setPhoto(self.image)

    def getPenColor(self):
        return self.__penColor.name()

    def setPhoto(self, image=None):

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = self.fitImage(image)
            image = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            self.__board = QPixmap.fromImage(image)
            self.__board.scaled(self.width(), self.height(), Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.setPixmap(self.__board)
        else:
            QImageTemp = self.__board.toImage()
            QImageTemp = QImageTemp.convertToFormat(QImage.Format_Grayscale8)
            # QImageTemp = self.fitImage(QImageTemp)
            self.__board = QPixmap.fromImage(QImageTemp)
            self.__board.scaled(self.width(), self.height(), Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.setPixmap(self.__board)





    def changeDraw(self,draw):
        self.drawTool=draw

    def fitImage(self,image):

        width = image.shape[1]
        height = image.shape[0]
        labelWidth = self.width()
        labelHeight = self.height()
        ratio = height / width
        new_width = width
        new_height = height

        if height > labelHeight or width > labelWidth:
            if ratio < 1:
                new_width = labelWidth
                new_height = int(new_width * ratio)
            else:
                new_height = labelHeight
                new_width = int(new_height * (width / height))
        image = cv2.resize(image, (new_width, new_height))
        self.resize(new_width, new_height)
        self.W,self.H = new_width, new_height

        self.scale = height / new_height

        print('resize ' ,image.shape)
        return image

    def paintEvent(self, paintEvent):

        if self.draw and not self.endPoint.isNull() and not self.lastPoint.isNull():
            x = self.lastPoint.x()
            y = self.lastPoint.y()
            if self.drawTool !='Line':
                w = self.endPoint.x() - x
                h = self.endPoint.y() - y
            else:
                w = self.endPoint.x()
                h = self.endPoint.y()

            self.temp = self.__board.copy()
            pp = QPainter(self.temp)
            pen = QPen(self.__penColor, self.__thickness)
            pen.setStyle(self.__penStyle)
            pp.setPen(pen)
            if self.drawTool=='Rectangle':
                pp.drawRect(x, y, w, h)
            elif self.drawTool == "Elipse":
                pp.drawEllipse(x, y, w, h)
            elif self.drawTool == "Pie":
                pp.drawPie(x, y, w, h,0*16,120*16)
            elif self.drawTool =='Line':
                pp.drawLine(x, y, w, h)
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.temp)
            self.tmpImage = self.convertQImageToMat(self.temp.toImage())

        elif self.crop and not self.endPoint.isNull() and not self.lastPoint.isNull():
            x = self.lastPoint.x()
            y = self.lastPoint.y()

            w = self.endPoint.x() - x
            h = self.endPoint.y() - y

            self.temp = self.__board.copy()
            pp = QPainter(self.temp)

            pp.drawRect(x, y, w, h)

            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.temp)
            self.tmpImage = self.convertQImageToMat(self.temp.toImage())
        else:


            painter = QPainter(self)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(0, 0, self.__board)
            self.tmpImage = self.convertQImageToMat(self.__board.toImage())


    def mousePressEvent(self, mouseEvent):
        if mouseEvent.buttons() == Qt.LeftButton and (self.draw or self.crop):

            if self.drawTool == 'brush' and self.draw:
                self.lastPoint = mouseEvent.pos()
                self.endPoint = self.lastPoint
            else:
                self.lastPoint = mouseEvent.pos()


    def mouseMoveEvent(self, mouseEvent):

        if mouseEvent.buttons() == Qt.LeftButton and (self.draw or self.crop):
            if self.drawTool == 'brush' and self.draw:
                self.lastPoint = mouseEvent.pos()
                pen = QPen(self.__penColor, self.__thickness)
                pen.setStyle(self.__penStyle)
                self.__painter.begin(self.__board)
                self.__painter.setPen(pen)
                self.__painter.drawLine(self.lastPoint, self.endPoint)
                self.__painter.end()
                self.endPoint = self.lastPoint
            else:
                self.endPoint = mouseEvent.pos()
            self.update()

    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.1
        else:  # 滚轮下滚
            self.scale *= 0.9
        self.adjustSize()
        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        if mouseEvent.button() == Qt.LeftButton:
            if self.draw:

                self.endPoint = mouseEvent.pos()
                self.endPoint = QPoint()
                self.lastPoint =QPoint()
                self.update()
                self.__board = self.temp.copy()

            elif self.crop:
                self.endPoint = mouseEvent.pos()
                self.temp = self.__board.copy(QRect(self.lastPoint,self.endPoint))
                self.endPoint = QPoint()
                self.lastPoint = QPoint()
                self.update()
                self.__board = self.temp.copy()



    def convertQImageToMat(self, image):
        '''  Converts a QImage into an opencv MAT format  '''

        image = image.convertToFormat(4)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        arr = cv2.cvtColor(arr,cv2.COLOR_BGRA2BGR)
        return arr
        # incomingImage = image.convertToFormat(QImage.Format_RGBX8888)
        # ptr = incomingImage.constBits()
        # ptr.setsize(incomingImage.byteCount())
        # cv_im_in = np.array(ptr, copy=True).reshape(incomingImage.height(), incomingImage.width(), 4)
        # cv_im_in = cv2.cvtColor(cv_im_in, cv2.COLOR_BGRA2BGR)
        # return cv_im_in
    def getBoard(self):
        return self.__board
    def getRatio(self):
        return self.ratio
    def ChangePenColor(self, color="black"):
        # 改变画笔颜色
        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):
        # 改变画笔粗细
        self.__thickness = thickness

    def ChangePenStyle(self,style):
        self.__penStyle = style

    def comparePressedImage(self):
        self.setPhoto(self.oriImage)

    def compareReleasedImage(self):
        self.setPhoto(self.image)

class tabWidget(QWidget):
    def __init__(self, Parent=None,image=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        self.VLayout = QVBoxLayout(self)
        self.imageLabel = PaintBoard(Parent=self,image=image)
        self.VLayout.addWidget(self.imageLabel)
        # self.HLayout = QHBoxLayout(self)
        # self.compareButton = QPushButton(parent=self,text='Compared')
        # self.compareButton.setMaximumSize(QtCore.QSize(200, 50))
        # self.compareButton.setMinimumSize(QtCore.QSize(200, 50))
        # self.compareButton.pressed.connect(self.comparePressedImage)
        # self.compareButton.released.connect(self.compareReleasedImage)
        # self.compareButton.setStyleSheet("QPushButton{border:2px solid; background-color:rgb(255, 255, 255)}"
        #                                  "QPushButton:pressed{border:2px solid; background-color:rgb(0, 0, 0);color:rgb(255,255,255)}"
        #                                 )
        # self.HLayout.addWidget(self.compareButton)
        # self.VLayout.addLayout(self.HLayout)

    # def comparePressedImage(self):
    #     self.imageLabel.setPhoto(self.imageLabel.oriImage)
    #
    # def compareReleasedImage(self):
    #     self.imageLabel.setPhoto(self.imageLabel.image)


