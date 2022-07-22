from PyQt5.QtWidgets import QWidget,QApplication, QPushButton,QMainWindow,QFileDialog,QColorDialog,QMessageBox
from PyQt5 import uic,QtGui
from PyQt5.QtGui import QImage,QIntValidator
from PyQt5.QtCore import Qt, QSize, QRect,QEvent
from PyQt5 import QtCore, QtGui, QtWidgets
from testUI import Ui_MainWindow
import cv2
import numpy as np
from PIL import Image
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PIL.ImageFilter import (
    CONTOUR, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
    EMBOSS, FIND_EDGES
)
from scipy.interpolate import UnivariateSpline
from ImageLabel import PaintBoard,tabWidget

class Stats(QMainWindow):
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit

        # one
        # self.ui = uic.loadUi("testUI.ui")
        # two
        super().__init__()
        self.historyImage = []
        # Added code here
        self.filename = None  # Will hold the image address location
        self.tmpImage = None  # Will hold the temporary image for display
        self.brightness_value_now = 0  # Updated brightness value
        self.blur_value_now = 0  # Updated blur value
        self.saturation_value_now = 0
        self.hue_value_now= 0
        self.dilate_value_now = 0
        self.erode_value_now = 0
        self.thresh_value_now = 0
        self.maxval_value_now = 0
        self.threshType = cv2.THRESH_TOZERO


        self.ui = Ui_MainWindow()
        # 初始化界面
        self.ui.setupUi(self)
        self.loadImage()

        # # 信号和槽
        self.ui.canvasTabWidget.tabCloseRequested.connect(self.tabClose)
        self.ui.canvasTabWidget.currentChanged.connect(self.tabChange)
        self.ui.actionNew.triggered.connect(self.loadImage)
        self.ui.actionUndo.triggered.connect(self.undoImage)
        self.ui.actionOriginalImage.triggered.connect(self.setOriginalImage)
        self.ui.cropButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.drawButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.filterButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.effectButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))

        # filter function
        self.ui.brightnessHorizontalSlider.valueChanged['int'].connect(self.brightness_value)
        self.ui.blurHorizontalSlider.valueChanged['int'].connect(self.blur_value)
        self.ui.saturationHorizontalSlider.valueChanged['int'].connect(self.saturation_value)
        self.ui.hueHorizontalSlider.valueChanged['int'].connect(self.hue_value)
        self.ui.dilateHorizontalSlider.valueChanged['int'].connect(self.dilate_value)
        self.ui.erodeHorizontalSlider.valueChanged['int'].connect(self.erode_value)
        self.ui.threshHorizontalSlider.valueChanged['int'].connect(self.thresh_value)
        self.ui.maxvalHorizontalSlider.valueChanged['int'].connect(self.maxval_value)

        self.ui.brightnessLineEdit.setValidator(QIntValidator(0, 100, self))
        self.ui.brightnessLineEdit.textChanged.connect(self.brightness_value)

        self.ui.blurLineEdit.setValidator(QIntValidator(0, 100, self))
        self.ui.blurLineEdit.textChanged.connect(self.blur_value)

        self.ui.dilateLineEdit.setValidator(QIntValidator(0, 100, self))
        self.ui.dilateLineEdit.textChanged.connect(self.dilate_value)

        self.ui.erodeLineEdit.setValidator(QIntValidator(0, 100, self))
        self.ui.erodeLineEdit.textChanged.connect(self.erode_value)

        self.ui.saturationLineEdit.setValidator(QIntValidator(-100, 100, self))
        self.ui.saturationLineEdit.textChanged.connect(self.saturation_value)

        self.ui.hueLineEdit.setValidator(QIntValidator(-100, 100, self))
        self.ui.hueLineEdit.textChanged.connect(self.hue_value)

        self.ui.threshLineEdit.setValidator(QIntValidator(-100, 100, self))
        self.ui.threshLineEdit.textChanged.connect(self.thresh_value)
        self.ui.maxvalLineEdit.setValidator(QIntValidator(-100, 100, self))
        self.ui.maxvalLineEdit.textChanged.connect(self.maxval_value)
        self.ui.threshComboBox.currentIndexChanged.connect(self.changeThreshType)
        self.ui.thresholdResetButton.clicked.connect(self.threholdReset)
        self.ui.filterApplyButton.clicked.connect(self.applyEffect)

        # crop function
        self.ui.widthLineEdit.setValidator(QIntValidator(1, 5000, self))
        self.ui.heightLineEdit.setValidator(QIntValidator(1, 5000, self))
        self.ui.sizePushButton.clicked.connect(self.changSize)
        self.ui.rotateLeft.clicked.connect(lambda: self.rotate('LEFT'))
        self.ui.rotateRight.clicked.connect(lambda: self.rotate('RIGHT'))
        self.ui.horizontalPushButton.clicked.connect(lambda: self.flip('HORIZONTAL'))
        self.ui.verticalPushButton.clicked.connect(lambda: self.flip('VERTICAL'))

        # draw function
        self.ui.colorPickPushButton.clicked.connect(self.colorPick)
        self.ui.penSizeHorizontalSlider.valueChanged['int'].connect(self.changePenSize)
        self.ui.penSizeLineEdit.setValidator(QIntValidator(0, 25, self))
        self.ui.penSizeLineEdit.textChanged.connect(self.changePenSize)
        self.ui.penStyleComboBox.currentTextChanged.connect(self.changePenStyle)
        self.ui.drawApplyPushButton.clicked.connect(self.applyEffect)

        self.ui.brushRadioButton.clicked.connect(lambda :self.changeDraw(draw='brush'))
        self.ui.rectRadioButton.clicked.connect(lambda :self.changeDraw(draw='Rectangle'))
        self.ui.pieRadioButton.clicked.connect(lambda :self.changeDraw(draw='Pie'))
        self.ui.eplipseRadioButton.clicked.connect(lambda :self.changeDraw(draw='Elipse'))
        self.ui.lineRadioButton.clicked.connect(lambda :self.changeDraw(draw='Line'))

        # effect function
        self.ui.brightnessPushButton.clicked.connect(lambda:self.changeEffect(effect='Brightness'))
        self.ui.hdrPushButton.clicked.connect(lambda:self.changeEffect(effect='HDR'))
        self.ui.greyScalePushButton.clicked.connect(lambda:self.changeEffect(effect='Greyscale'))
        self.ui.darkerPushButton.clicked.connect(lambda:self.changeEffect(effect='Darker'))
        self.ui.sharpPushButton.clicked.connect(lambda:self.changeEffect(effect='Sharp'))
        self.ui.summerPushButton.clicked.connect(lambda:self.changeEffect(effect='Summer'))
        self.ui.winterPushButton.clicked.connect(lambda:self.changeEffect(effect='Winter'))
        self.ui.sepiaPushButton.clicked.connect(lambda:self.changeEffect(effect='Sepia'))
        self.ui.foilPushButton.clicked.connect(lambda:self.changeEffect(effect='Foil'))
        self.ui.pencilColorPushButton.clicked.connect(lambda:self.changeEffect(effect='PencilColor'))
        self.ui.invertPushButton.clicked.connect(lambda:self.changeEffect(effect='Invert'))
        self.ui.gaussianBlurPushButton.clicked.connect(lambda:self.changeEffect(effect='Gaussian'))
        self.ui.medianBlurPushButton.clicked.connect(lambda:self.changeEffect(effect='Median'))
        self.ui.originalPushButton.clicked.connect(lambda:self.changeEffect(effect='Original'))
        self.ui.effectApplyButton.clicked.connect(self.applyEffect)

        #     ai
        self.ui.backRemovePushButton.clicked.connect(self.removeBackground)
    # ai filter
    def removeBackground(self):


    def undoImage(self):
        self.currentImage.popHistory()



    def tabChange(self):

        print('change to {} tab'.format(self.ui.canvasTabWidget.currentIndex()))
        self.currentImage = self.ui.canvasTabWidget.currentWidget().imageLabel

    def tabClose(self):
        self.ui.canvasTabWidget.removeTab(self.ui.canvasTabWidget.currentIndex())
    def changeEvent(self, event):

        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                print('changeEvent: Minimised')
            elif event.oldState() & Qt.WindowMinimized:
                print('changeEvent: Normal/Maximised/FullScreen')

    def testTab(self,image):

        tempWidget = tabWidget(image = image)
        index = self.filename.rfind('/')
        name = self.filename[index+1:]
        self.ui.canvasTabWidget.addTab(tempWidget, name)
        self.ui.canvasTabWidget.setCurrentIndex(len(self.ui.canvasTabWidget)-1)
        self.currentImage = self.ui.canvasTabWidget.currentWidget().imageLabel
        self.currentImage.addHistory(self.currentImage.image)
        self.currentImage.setFileName(self.filename)
        color = self.currentImage.getPenColor()
        self.ui.colorPickPushButton.setStyleSheet('background-color: {};'.format(color))


    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Image","", "JPG Files (*.jpeg *.jpg );;PNG Files (*.png)")
        if self.filename:
            self.oriImage = cv2.imread(self.filename)
            # self.oriImage = cv2.cvtColor(self.oriImage, cv2.COLOR_BGR2RGB)
            self.image = self.oriImage.copy()
            self.tmpImage = self.oriImage.copy()
            # -----------------------------------------------------------

            image = cv2.imread(self.filename)
            self.testTab(image)

            # -----------------------------------------------------------
            # self.setPhoto(self.image)

        else:
            QMessageBox.information(self, "Error",
                                    "Unable to open image.", QMessageBox.Ok)

        # self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]

    # def setPhoto(self, image=None, gray=False):
    #     """
    #         This function will take image input and resize it
    #         only for display purpose and convert it to QImage
    #         to set at the label.
    #     """
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     #         # # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     #         # gray = True
    #     image = self.fitImage(image)
    #     # if gray:
    #     #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     #     image = QImage(image, image.shape[1], image.shape[0],image.shape[1],QImage.Format_Indexed8)
    #     # else:
    #     image = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
    #     # image_show = QtGui.QPixmap.fromImage(image)
    #     # self.imageLabel.setPixmap(image_show)
    #     # self.ui.imageLabel.setCursor(Qt.CrossCursor)


    def setOriginalImage(self):
        self.currentImage.image = self.currentImage.oriImage
        self.currentImage.setPhoto(self.currentImage.image)

    # def fitImage(self,image):
    #
    #     width = image.shape[1]
    #     height = image.shape[0]
    #     labelWidth = self.ui.imageLabel.width()
    #     labelHeight = self.ui.imageLabel.height()
    #     ratio = height / width
    #     new_width = width
    #     new_height = height
    #
    #     if height > labelHeight or width > labelWidth:
    #         if ratio < 1:
    #             new_width = labelWidth
    #             new_height = int(new_width * ratio)
    #         else:
    #             new_height = labelHeight
    #             new_width = int(new_height * (width / height))
    #     image = cv2.resize(image, (new_width, new_height))
    #
    #     self.ratio = height / new_height
    #
    #     # if width / labelWidth >= height / labelHeight:  ##比较图片宽度与label宽度之比和图片高度与label高度之比
    #     #     ratio = width / labelWidth
    #     # else:
    #     #     ratio = height / labelHeight
    #     #
    #     # new_width = int(width / ratio)
    #     # new_height = int(height / ratio)
    #     # image = cv2.resize(image,(new_width,new_height))
    #     return image

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def applyEffect(self):
        self.currentImage.image =self.currentImage.tmpImage
        self.currentImage.addHistory(self.currentImage.image)
    def changeDraw(self,draw='brush'):
        self.currentImage.changeDraw(draw)

    def colorPick(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.currentImage.ChangePenColor(color=color.name())
            self.ui.colorPickPushButton.setStyleSheet('background-color: {};'.format(color.name()))

        # self.currentImage.__penColor = QColor('Yellow')
        # if self.currentImage.__penColor.isValid():
        #     print(self.currentImage.__penColor.name())
    def changePenSize(self,size):
        if size != '' and int(size) >= 0 and int(size)<=25:
            self.currentImage.ChangePenThickness(thickness=int(size))
            self.ui.penSizeLineEdit.blockSignals(True)
            self.ui.penSizeHorizontalSlider.blockSignals(True)

            self.ui.penSizeLineEdit.setText(str(size))
            self.ui.penSizeHorizontalSlider.setSliderPosition(int(size))

            self.ui.penSizeLineEdit.blockSignals(False)
            self.ui.penSizeHorizontalSlider.blockSignals(False)
    def changePenStyle(self):
        index = self.ui.penStyleComboBox.currentIndex()
        if index == 0:
            style = Qt.SolidLine
        elif index == 1:
            style = Qt.DashLine
        elif index == 2:
            style =Qt.DotLine
        self.currentImage.ChangePenStyle(style)

    def changSize (self):
        if self.ui.widthLineEdit is not None and self.ui.heightLineEdit is not None:
            new_width = int(self.ui.widthLineEdit.text())
            new_height = int(self.ui.heightLineEdit.text())
            if new_width >0 and new_height>0:
                self.currentImage.image = cv2.resize(self.currentImage.image,(new_width,new_height))
                self.currentImage.addHistory(self.currentImage.image)
                self.currentImage.setPhoto(self.currentImage.image)

    def LookupTable(self,x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))

    def changeEffect(self,effect='Original'):

        img = self.currentImage.image.copy()
        if effect == 'Original':
            img = self.currentImage.image.copy()
        elif effect == 'GreyScale':
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # self.tmpImage = img
            # self.setPhoto(self.tmpImage,True)
            return
        elif effect == 'Brightness':
            img = cv2.convertScaleAbs(img, beta=50)
        elif effect == 'Darker':
            img = cv2.convertScaleAbs(img, beta=-50)
        elif effect =='Sharp':
            kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
        elif effect == 'Sepia':
            img = np.array(img, dtype=np.float64)  # converting to float to prevent loss
            img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                                            [0.349, 0.686, 0.168],
                                                            [0.393, 0.769,
                                                             0.189]]))  # multipying image with special sepia matrix
            img[np.where(img > 255)] = 255  # normalizing values greater than 255 to 255
            img = np.array(img, dtype=np.uint8)

        elif effect == 'Foil':
            img = np.array(Image.fromarray(img).filter(EMBOSS))
        elif effect == 'Median':
            img = cv2.medianBlur(img, 41)
        elif effect == 'Gaussian':
            img = cv2.GaussianBlur(img, (41, 41), 0)
        elif effect=='PencilColor':
            sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
            img = sk_color
        elif effect =='HDR':
            img = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        elif effect =='Invert':
            img = cv2.bitwise_not(img)
        elif effect =='Summer':
            # img = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
            # kernel = np.array([[0.272, 0.534, 0.131],
            #                    [0.349, 0.686, 0.168],
            #                    [0.393, 0.769, 0.189]])
            # img = cv2.transform(img, kernel)
            # img[np.where(img > 255)] = 255  # normalizing
            increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
            decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
            blue_channel, green_channel, red_channel = cv2.split(img)
            red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
            blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
            img = cv2.merge((blue_channel, green_channel, red_channel))
        elif effect =='Winter':

            increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
            decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
            blue_channel, green_channel, red_channel = cv2.split(img)
            red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
            blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
            img = cv2.merge((blue_channel, green_channel, red_channel))
        self.currentImage.tmpImage = img
        self.currentImage.setPhoto(self.currentImage.tmpImage)

    # defining a function

    def thresh_value(self,value):
        if value != '' and int(value) in range(0, 256):
            self.ui.threshHorizontalSlider.blockSignals(True)
            self.ui.threshLineEdit.blockSignals(True)

            self.ui.threshLineEdit.setText(str(value))
            self.ui.threshHorizontalSlider.setSliderPosition(int(value))

            self.ui.threshHorizontalSlider.blockSignals(False)
            self.ui.threshLineEdit.blockSignals(False)

            self.thresh_value_now = int(value)
            print('thresh: ', value)
            self.update()

    def maxval_value(self,value):
        if value != '' and int(value) in range(0, 256):
            self.ui.maxvalHorizontalSlider.blockSignals(True)
            self.ui.maxvalLineEdit.blockSignals(True)

            self.ui.maxvalLineEdit.setText(str(value))
            self.ui.maxvalHorizontalSlider.setSliderPosition(int(value))

            self.ui.maxvalHorizontalSlider.blockSignals(False)
            self.ui.maxvalLineEdit.blockSignals(False)

            self.maxval_value_now = int(value)
            print('maxval: ', value)
            self.update()
    def threholdReset(self):
        self.ui.threshHorizontalSlider.blockSignals(True)
        self.ui.maxvalHorizontalSlider.blockSignals(True)
        self.ui.threshLineEdit.blockSignals(True)
        self.ui.maxvalLineEdit.blockSignals(True)
        self.ui.threshLineEdit.setText('0')
        self.ui.maxvalLineEdit.setText('0')
        self.ui.threshHorizontalSlider.setSliderPosition(0)
        self.ui.maxvalHorizontalSlider.setSliderPosition(0)
        self.ui.threshHorizontalSlider.blockSignals(False)
        self.ui.maxvalHorizontalSlider.blockSignals(False)
        self.ui.threshLineEdit.blockSignals(False)
        self.ui.maxvalLineEdit.blockSignals(False)
        self.thresh_value_now = 0
        self.maxval_value_now = 0
        self.ui.threshComboBox.setCurrentIndex(0)


    def changeThreshType(self):
        index = self.ui.threshComboBox.currentIndex()
        if index == 2:
            self.threshType = cv2.THRESH_BINARY
        elif index == 3:
            self.threshType = cv2.THRESH_BINARY_INV
        elif index == 4:
            self.threshType = cv2.THRESH_TRUNC
        elif index == 0:
            self.threshType = cv2.THRESH_TOZERO
        elif index == 1:
            self.threshType = cv2.THRESH_TOZERO_INV
        self.update()

    def dilate_value(self,value):

        if value != '' and int(value) in range(0, 101):
            self.ui.dilateHorizontalSlider.blockSignals(True)
            self.ui.dilateLineEdit.blockSignals(True)

            self.ui.dilateLineEdit.setText(str(value))
            self.ui.dilateHorizontalSlider.setSliderPosition(int(value))

            self.ui.dilateHorizontalSlider.blockSignals(False)
            self.ui.dilateLineEdit.blockSignals(False)

            self.dilate_value_now = int(value)
            print('dilate: ', value)
            self.update()
    def erode_value(self,value):

        if value != '' and int(value) in range(0, 101):
            self.ui.erodeHorizontalSlider.blockSignals(True)
            self.ui.erodeLineEdit.blockSignals(True)

            self.ui.erodeLineEdit.setText(str(value))
            self.ui.erodeHorizontalSlider.setSliderPosition(int(value))

            self.ui.erodeHorizontalSlider.blockSignals(False)
            self.ui.erodeLineEdit.blockSignals(False)

            self.erode_value_now = int(value)
            print('erode: ', value)
            self.update()

    def brightness_value(self, value):
        """ This function will take value from the slider
            for the brightness from 0 to 99
        """
        if value != '' and int(value) in range(-100, 101):
            self.ui.brightnessHorizontalSlider.blockSignals(True)
            self.ui.brightnessLineEdit.blockSignals(True)

            self.ui.brightnessLineEdit.setText(str(value))
            self.ui.brightnessHorizontalSlider.setSliderPosition(int(value))

            self.ui.brightnessHorizontalSlider.blockSignals(False)
            self.ui.brightnessLineEdit.blockSignals(False)

            self.brightness_value_now = int(value)
            print('Brightness: ', value)
            self.update()

    def saturation_value(self,value):
        if value != '' and int(value) in range(-100, 101):

            self.ui.saturationHorizontalSlider.blockSignals(True)
            self.ui.saturationLineEdit.blockSignals(True)

            self.ui.saturationLineEdit.setText(str(value))
            self.ui.saturationHorizontalSlider.setSliderPosition(int(value))

            self.ui.saturationHorizontalSlider.blockSignals(False)
            self.ui.saturationLineEdit.blockSignals(False)

            self.saturation_value_now = int(value)
            print('saturation: ', value)
            self.update()
    def hue_value(self,value):
            if value != '' and int(value) in range(-100, 101):

                self.ui.hueHorizontalSlider.blockSignals(True)
                self.ui.hueLineEdit.blockSignals(True)

                self.ui.hueLineEdit.setText(str(value))
                self.ui.hueHorizontalSlider.setSliderPosition(int(value))

                self.ui.hueHorizontalSlider.blockSignals(False)
                self.ui.hueLineEdit.blockSignals(False)

                self.hue_value_now = int(value)
                print('hue: ', value)
                self.update()


    def blur_value(self, value):
        """ This function will take value from the slider
            for the blur from 0 to 99 """
        if value != '' and int(value) in range(0, 101):

            self.ui.blurHorizontalSlider.blockSignals(True)
            self.ui.blurLineEdit.blockSignals(True)

            self.ui.blurLineEdit.setText(str(value))
            self.ui.blurHorizontalSlider.setSliderPosition(int(value))

            self.ui.blurHorizontalSlider.blockSignals(False)
            self.ui.blurLineEdit.blockSignals(False)

            self.blur_value_now = int(value)
            print('Blur: ', value)
            self.update()
    # details filter
    def changeDilate(self,img,value):

        kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
        img = cv2.dilate(img, kernel_size,iterations=1)
        return img
    def changeErode(self,img,value):
        kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
        img = cv2.erode(img, kernel_size)
        return img

    def changeBlur(self, img, value):
        """ This function will take the img image and blur values as inputs.
            After perform blur operation using opencv function, it returns
            the image img.
        """
        kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
        img = cv2.blur(img, kernel_size)
        return img

    def changeThreshold(self,img,thresh,maxval):

        ori_img,img = cv2.threshold(img, thresh, maxval,self.threshType)
        return img

    def changeBrightness(self, img, value):
        """
            This function will take an image (img) and the brightness
            value. It will perform the brightness change using OpenCv
            and after split, will merge the img and return it.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.int16(v)
        v+=value
        v= np.clip(v,0,255)
        # if value >0:
        #     lim = 255 - value
        #     v[v > lim] = 255
        #     v[v <= lim] += value
        # else:
        #     lim = 0 - value
        #     v[v >= lim] += value
        #     v[v < lim] == 0
        v = np.uint8(v)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    def changeHue(self, img, value):
        """
            This function will take an image (img) and the brightness
            value. It will perform the brightness change using OpenCv
            and after split, will merge the img and return it.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = np.int16(h)
        h+=value
        h = np.clip(h,0,255)
        h = np.uint8(h)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img



    def changeSaturation(self, img, value):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.int16(s)
        s+=value
        s = np.clip(s,0,255)
        s = np.uint8(s)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    def rotate(self, option):
        h, w, _ = self.currentImage.image.shape
        temp = np.zeros((w, h, 3), np.uint8)  # null image with inverted dimensions
        if option == 'LEFT':
            for i in range(0, w):
                temp[w - i - 1, :, :] = self.currentImage.image[:, i, :]
        elif option == 'RIGHT':
            for j in range(0, h):
                temp[:, h - j - 1, :] = self.currentImage.image[j, :, :]

        self.currentImage.image = temp
        self.currentImage.tmpImage = temp
        self.currentImage.setPhoto(self.currentImage.image)
        self.currentImage.addHistory(self.currentImage.image.copy())

    def flip(self, option):
        h, w, _ = self.currentImage.image.shape
        temp = np.zeros((h, w, 3), np.uint8)
        if option == 'HORIZONTAL':
            for i in range(0, w):
                temp[:, i, :] = self.currentImage.image[:, w - i - 1, :]
        elif option == 'VERTICAL':
            for j in range(0, h):
                temp[j, :, :] = self.currentImage.image[h - j - 1, :, :]
        self.currentImage.image = temp
        self.currentImage.tmpImage = temp
        self.currentImage.setPhoto(self.currentImage.image)
        self.currentImage.addHistory(self.currentImage.image.copy())

    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        self.currentImage.tmpImage = self.changeBrightness(self.currentImage.image, self.brightness_value_now)
        self.currentImage.tmpImage = self.changeBlur(self.currentImage.tmpImage, self.blur_value_now)
        self.currentImage.tmpImage = self.changeSaturation(self.currentImage.tmpImage, self.saturation_value_now)
        self.currentImage.tmpImage = self.changeHue(self.currentImage.tmpImage, self.hue_value_now)
        self.currentImage.tmpImage = self.changeDilate(self.currentImage.tmpImage, self.dilate_value_now)
        self.currentImage.tmpImage = self.changeErode(self.currentImage.tmpImage, self.erode_value_now)
        self.currentImage.tmpImage = self.changeThreshold(self.currentImage.tmpImage, self.thresh_value_now,self.maxval_value_now)
        self.currentImage.setPhoto(self.currentImage.tmpImage)


    def handleCalc(self,text):

        print(self.ui.penStyleComboBox.currentIndex())

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self.currentImage.setPhoto(self.currentImage.image)


app = QApplication([])
stats = Stats()
stats.show()
app.exec_()

