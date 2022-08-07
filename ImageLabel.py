from PyQt5.QtGui import *


import numpy as np

from PyQt5.Qt import QPainter, QPoint, QPen
from PyQt5.QtCore import Qt, QRect
from PyQt5.Qt import QWidget, QColor, QPixmap,  QSize
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel, QMessageBox
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
        self.setAlignment(Qt.AlignCenter)

    def __InitData(self,image):
        #
        self.filename = None
        self.Line_list = [0,0,0,0]
        self.Point_list = []
        self.Elipse_list = [0,0,0,0]
        self.Rectangle_list = [0,0,0,0]
        self.Polygon_list = []
        self.Pie_list = [0,0,0,0]
        self.Arc_list = [0,0,0,0]
        self.drawTool = 'brush'
        self.draw = False
        self.imageHistory = []
        self.redoHistory = []
        #

        self.oriImage = image.copy()
        self.image = image.copy()
        self.tmpImage = image.copy()
        self.setPhoto(self.image)
        self.shape = self.image.shape

        self.__IsEmpty = True  # 默认为空画板
        self.EraserMode = False  # 默认为禁用橡皮擦模式

        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.__painter = QPainter()  # 新建绘图工具
        self.__thickness = 5  # 默认画笔粗细为10px
        self.__penColor = QColor("black")  # 设置默认画笔颜色为黑色
        self.__penStyle = Qt.SolidLine

        self.__colorList = QColor.colorNames()  # 获取颜色列表

    def setFileName(self,name):
        self.filename = name

    def setDraw(self,draw):
        self.draw =draw
        if draw :
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


    def popHistory(self):
        if len(self.imageHistory) > 1:
            self.redoHistory.append(self.imageHistory.pop())
            self.image = self.imageHistory[-1]
            self.setPhoto(self.image)

    def getPenColor(self):
        return self.__penColor.name()

    def setPhoto(self, image=None):
        """
            This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        try:
            if len(image.shape) ==3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.fitImage(image)
                image = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
                self.__board = QPixmap.fromImage(image)
                # self.__board.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
                self.setPixmap(self.__board)
            else:
                QImageTemp = self.__board.toImage()
                QImageTemp = QImageTemp.convertToFormat(QImage.Format_Grayscale8)
                self.__board = QPixmap.fromImage(QImageTemp)
                # self.__board.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
                self.setPixmap(self.__board)

        except:
            QMessageBox.information(self, "Error", QMessageBox.Ok)



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

        self.ratio = height / new_height

        print('resize ' ,image.shape)
        return image




    def paintEvent(self, paintEvent):
        # 绘图事件
        # 绘图时必须使用QPainter的实例，此处为__painter
        # 绘图在begin()函数与end()函数间进行
        # begin(param)的参数要指定绘图设备，即把图画在哪里
        # drawPixmap用于绘制QPixmap类型的对象
        # pass
        # if self.is_start_cut and not self.is_midbutton:
            # print(self.start_point, self.current_point)

        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0, 0,self.__board)

        self.__painter.end()

        # rect = QRect(self.Rectangle_list[0], self.Rectangle_list[1], self.Rectangle_list[2], self.Rectangle_list[3])
        # new_pixmap = self.__board.copy(rect)
        # new_pixmap.save(r'test.png')

        #
        self.tmpImage = self.convertQImageToMat(self.__board.toImage())

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        if mouseEvent.buttons() == Qt.LeftButton and self.draw:
            if self.drawTool == 'brush':
                self.__currentPos = mouseEvent.pos()
                self.__lastPos = self.__currentPos
            elif self.drawTool == "Line":
                self.Line_list[0] = mouseEvent.x()
                self.Line_list[1] = mouseEvent.y()
                print("start", self.Line_list[0], self.Line_list[1])
            elif self.drawTool == "Point":
                self.Point_list.append(mouseEvent.x())
                self.Point_list.append(mouseEvent.y())
                self.update()
            elif self.drawTool == "Elipse":
                self.Elipse_list[0] = mouseEvent.x()
                self.Elipse_list[1] = mouseEvent.y()
                print("start", self.Elipse_list[0], self.Elipse_list[1])
            elif self.drawTool == "Rectangle":
                self.Rectangle_list[0] = mouseEvent.x()
                self.Rectangle_list[1] = mouseEvent.y()
                print("start", self.Rectangle_list[0], self.Rectangle_list[1])
            elif self.drawTool == "Polygon":
                self.Polygon_list.append(mouseEvent.x())
                self.Polygon_list.append(mouseEvent.y())
                print(self.Polygon_list)
                self.update()
            elif self.drawTool == "Pie":
                self.Pie_list[0] = mouseEvent.x()
                self.Pie_list[1] = mouseEvent.y()
            elif self.drawTool == "Arc":
                self.Arc_list[0] = mouseEvent.x()
                self.Arc_list[1] = mouseEvent.y()

            self.update()

    def mouseMoveEvent(self, mouseEvent):

        if mouseEvent.buttons() == Qt.LeftButton and self.draw:
            if self.drawTool == 'brush':
                # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
                self.__currentPos = mouseEvent.pos()
                pen = QPen(self.__penColor, self.__thickness)
                pen.setStyle(self.__penStyle)
                self.__painter.begin(self.__board)
                self.__painter.setPen(pen)  # 设置画笔颜色，粗细
                #
                # # 画线
                #
                self.__painter.drawLine(self.__lastPos, self.__currentPos)

                self.__painter.end()
                self.__lastPos = self.__currentPos
                self.update()

            self.__painter.begin(self.__board)

            self.__painter.end()
            self.update()  # 更新显示


    def mouseReleaseEvent(self, mouseEvent):
        if mouseEvent.button() == Qt.LeftButton and self.draw:
            if self.drawTool != 'brush':
                pen = QPen(self.__penColor, self.__thickness)
                pen.setStyle(self.__penStyle)
                self.__painter.begin(self.__board)
                self.__painter.setPen(pen)  # 设置画笔颜色，粗细
                if self.drawTool == "Line":
                    self.Line_list[2] = mouseEvent.x()
                    self.Line_list[3] = mouseEvent.y()
                    self.__painter.drawLine(self.Line_list[0],self.Line_list[1],self.Line_list[2],self.Line_list[3])
                    print("end", self.Line_list[2], self.Line_list[3])
                elif self.drawTool == "Elipse":
                    self.Elipse_list[2] = mouseEvent.x() - self.Elipse_list[0]
                    self.Elipse_list[3] = mouseEvent.y() - self.Elipse_list[1]
                    self.__painter.drawEllipse(self.Elipse_list[0],self.Elipse_list[1],self.Elipse_list[2],self.Elipse_list[3])
                    print("Radius", self.Elipse_list[2], self.Elipse_list[3])
                elif self.drawTool == "Rectangle":
                    self.Rectangle_list[2] = mouseEvent.x() - self.Rectangle_list[0]
                    self.Rectangle_list[3] = mouseEvent.y() - self.Rectangle_list[1]
                    self.__painter.drawRect(self.Rectangle_list[0],self.Rectangle_list[1],self.Rectangle_list[2],self.Rectangle_list[3])
                    print("Rectangle", self.Rectangle_list[2], self.Rectangle_list[3])
                elif self.drawTool == "Pie":
                    self.Pie_list[2] = mouseEvent.x() - self.Pie_list[0]
                    self.Pie_list[3] = mouseEvent.y() - self.Pie_list[1]
                    self.__painter.drawPie(self.Pie_list[0],self.Pie_list[1],self.Pie_list[2],self.Pie_list[3],0*16,120*16)
                elif self.drawTool == "Arc":
                    self.Arc_list[2] = mouseEvent.x() - self.Arc_list[0]
                    self.Arc_list[3] = mouseEvent.y() - self.Arc_list[1]
                    self.__painter.drawArc(self.Arc_list[0],self.Arc_list[1],self.Arc_list[2],self.Arc_list[3],30*16,120*16)

                self.__painter.end()
                self.update()

    def convertQImageToMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''


        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        arr = cv2.cvtColor(arr,cv2.COLOR_BGRA2BGR)
        return arr
    def getBoard(self):
        return self.__board
    def ChangePenColor(self, color="black"):
        # 改变画笔颜色
        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):
        # 改变画笔粗细
        self.__thickness = thickness

    def ChangePenStyle(self,style):
        self.__penStyle = style

class tabWidget(QWidget):
    def __init__(self, Parent=None,image=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        self.VLayout = QVBoxLayout(self)
        self.imageLabel = PaintBoard(Parent=self,image=image)
        self.VLayout.addWidget(self.imageLabel)
        self.compareButton = QPushButton(parent=self,text='Compared')
        self.compareButton.pressed.connect(self.comparePressedImage)
        self.compareButton.released.connect(self.compareReleasedImage)
        self.compareButton.setStyleSheet("border:2px solid; background-color:rgb(255, 255, 255);")
        self.VLayout.addWidget(self.compareButton)

    def comparePressedImage(self):
        self.imageLabel.setPhoto(self.imageLabel.oriImage)

    def compareReleasedImage(self):
        self.imageLabel.setPhoto(self.imageLabel.image)


