from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QColorDialog,QMessageBox
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, QEvent
from UI import Ui_MainWindow
import cv2
import numpy as np
from PIL import Image
from PIL.ImageFilter import EMBOSS
from scipy.interpolate import UnivariateSpline
from ImageCanvas import tabWidget
from rembg import remove
import tensorflow as tf
from Functions.AdvancedFunction import pixelate,kMeansImage,augContrast
from InProgress import ProgressBar
import time
from Functions import  EffectFunction,FilterFunction,CropRotateFunction


class Stats(QMainWindow):
    def __init__(self):

        super().__init__()
        self.historyImage = []
        self.draw =False
        self.crop =False
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
        self.contrast_value_now = 1
        self.threshType = cv2.THRESH_TOZERO
        self.ui = Ui_MainWindow()
        # 初始化界面
        self.ui.setupUi(self)
        self.loadImage()
        self.setupSign()
        self.showMaximized()

    def setupSign(self):
        # # 信号和槽
        self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W,self.currentImage.H))
        self.ui.comparePushButton.pressed.connect(self.currentImage.comparePressedImage)
        self.ui.comparePushButton.released.connect(self.currentImage.compareReleasedImage)
        self.ui.actionsave.triggered.connect(self.saveImage)
        self.ui.stackedWidget.currentChanged.connect(self.changeStacked)
        self.ui.canvasTabWidget.tabCloseRequested.connect(self.tabClose)
        self.ui.canvasTabWidget.currentChanged.connect(self.tabChange)
        self.ui.actionNew.triggered.connect(self.loadImage)
        self.ui.actionUndo.triggered.connect(self.undoImage)
        self.ui.actionRedo.triggered.connect(self.redoImage)
        self.ui.actionGet_Start.triggered.connect(self.help)
        self.ui.actionOriginalImage.triggered.connect(self.setOriginalImage)
        # self.ui.cropButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(5))
        self.ui.rotateButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.drawButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.filterButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.effectButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.advancedButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(4))
        # filter function
        self.ui.brightnessHorizontalSlider.valueChanged['int'].connect(self.brightness_value)
        self.ui.blurHorizontalSlider.valueChanged['int'].connect(self.blur_value)
        self.ui.saturationHorizontalSlider.valueChanged['int'].connect(self.saturation_value)
        self.ui.hueHorizontalSlider.valueChanged['int'].connect(self.hue_value)
        self.ui.dilateHorizontalSlider.valueChanged['int'].connect(self.dilate_value)
        self.ui.erodeHorizontalSlider.valueChanged['int'].connect(self.erode_value)
        self.ui.threshHorizontalSlider.valueChanged['int'].connect(self.thresh_value)
        self.ui.maxvalHorizontalSlider.valueChanged['int'].connect(self.maxval_value)
        self.ui.contraseHorizontalSlider.valueChanged['int'].connect(self.contrast_value)

        self.ui.contrastLineEdit.setValidator(QIntValidator(1, 10, self))
        self.ui.contrastLineEdit.textChanged.connect(self.contrast_value)

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


        # crop function
        self.ui.widthLineEdit.setValidator(QIntValidator(1, 5000, self))
        self.ui.heightLineEdit.setValidator(QIntValidator(1, 5000, self))
        self.ui.widthLineEdit.textChanged.connect(self.fixRatioWidth)
        self.ui.heightLineEdit.textChanged.connect(self.fixRatioHeight)
        self.ui.sizePushButton.clicked.connect(self.changeSize)
        self.ui.rotateLeft.clicked.connect(lambda: self.rotateFlip('LEFT'))
        self.ui.rotateRight.clicked.connect(lambda: self.rotateFlip('RIGHT'))
        self.ui.horizontalPushButton.clicked.connect(lambda: self.rotateFlip('HORIZONTAL'))
        self.ui.verticalPushButton.clicked.connect(lambda: self.rotateFlip('VERTICAL'))
        self.ui.cropPushButton.clicked.connect(self.changeCrop)

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
        # self.ui.effectApplyButton.clicked.connect(self.applyEffect)

        # advanced part
        self.ui.backRemovePushButton.clicked.connect(self.removeBackground)
        # self.ui.transferPushButton.clicked.connect(self.transferStyle)
        self.ui.autoContrastPushButton.clicked.connect(self.autoContrast)
        self.ui.pixel32PushButton.clicked.connect(lambda: self.changeToPiexl(size=32))
        self.ui.pixel48PushButton.clicked.connect(lambda: self.changeToPiexl(size=48))
        self.ui.pixel64PushButton.clicked.connect(lambda: self.changeToPiexl(size=64))
        self.ui.pixel128PushButton.clicked.connect(lambda: self.changeToPiexl(size=128))

    def help(self):
        import os
        os.startfile('Help.pdf')
        # return
    def fixRatioWidth(self,value):
        if self.ui.fixScaleCheckBox.isChecked():
            if value != '':
                width = int(value)
                height = int(width * self.currentImage.getRatio())
                # print(height)
                self.ui.heightLineEdit.blockSignals(True)
                self.ui.heightLineEdit.setText(str(height))
                self.ui.heightLineEdit.blockSignals(False)
    def fixRatioHeight(self,value):
        if self.ui.fixScaleCheckBox.isChecked():
            if value != '' :
                # print(1111)
                height = int(value)
                width = int(height/self.currentImage.getRatio())
                # print(width)
                self.ui.widthLineEdit.blockSignals(True)
                self.ui.widthLineEdit.setText(str(width))
                self.ui.widthLineEdit.blockSignals(False)
    def autoContrast(self):
        img = self.currentImage.image.copy()
        aug_img = augContrast(img)
        self.currentImage.tmpImage = aug_img
        self.currentImage.setPhoto(self.currentImage.tmpImage)

    def changeToPiexl(self,size):

        bar = ProgressBar(self)

        bar.set_value(0)
        img = self.currentImage.image.copy()
        time.sleep(0.1)
        bar.set_value(10)
        imgPiexl = pixelate(img, size, size)
        time.sleep(0.1)
        bar.set_value(50)
        newImage = kMeansImage(imgPiexl, 5)
        time.sleep(0.1)
        bar.set_value(80)
        self.currentImage.tmpImage = newImage
        self.currentImage.setPhoto(self.currentImage.tmpImage)
        time.sleep(0.1)
        bar.set_value(100)
        time.sleep(0.1)
        bar.close()
        QApplication.processEvents()


    def changeCrop(self):
        if self.crop == True:
            self.crop = False
            self.setCropButtonColor()

        else:
            self.crop = True
            self.setCropButtonColor()


        self.currentImage.setCrop(self.crop)
    def setCropButtonColor(self):
        if not self.crop:
            self.ui.cropPushButton.setStyleSheet("""
                        QPushButton{
                            border:2px solid rgb(255, 255, 255);
                            background-color:rgb(0, 0, 0);
                            color: rgb(255, 255, 255)
                        }
                        QPushButton:hover{

                            border: 2px solid rgb(0, 154, 206)
                        }
                        """)
        else:
            self.ui.cropPushButton.setStyleSheet("""
                                    QPushButton{
                                        border:2px solid rgb(0, 0, 0);
                                        background-color:rgb(255, 255, 255);
                                        color: rgb(0, 0, 0)
                                    }
                                    QPushButton:hover{

                                        border: 2px solid rgb(0, 154, 206)
                                    }
                                    """)


    def changeStacked(self):

        if self.ui.stackedWidget.currentIndex() == 1:
            self.draw = True
        else:
            self.draw = False
            self.crop = False
            self.setCropButtonColor()


        self.currentImage.setDraw(self.draw)
        self.currentImage.setCrop(self.crop)

    def tensor_to_image(self,tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        # img = PIL.Image.fromarray(tensor)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return np.squeeze(tensor, axis=0)

    def tf_load_img(self,path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def transferStyle(self):
        pass
        # filename, _ = QFileDialog.getOpenFileName(self, "Open Transfer Image", "",
        #                                                "JPG Files (*.jpeg *.jpg );;PNG Files (*.png)")
        # if filename:
        #     content_image = self.tf_load_img(self.filename)
        #     style_image =self.tf_load_img(filename)
        #     stylized_image = self.hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        #     # print(stylized_image.shape)
        #     # np.float32(stylized_image)
        #     self.currentImage.setPhoto(np.float32(stylized_image))


    def removeBackground(self):
        bar = ProgressBar(self)
        bar.set_value(0)
        time.sleep(0.1)
        img = self.currentImage.image.copy()
        bar.set_value(20)
        result = remove(img)
        time.sleep(0.1)
        bar.set_value(50)
        new_result = cv2.cvtColor(result,cv2.COLOR_BGRA2BGR)
        time.sleep(0.1)
        bar.set_value(80)
        self.currentImage.tmpImage = new_result
        self.currentImage.setPhoto(self.currentImage.tmpImage)
        time.sleep(0.1)
        bar.set_value(100)
        time.sleep(0.1)
        bar.close()

    def redoImage(self):
        self.currentImage.redo()

    def undoImage(self):
        self.currentImage.undo()



    def tabChange(self):

        # print('change to {} tab'.format(self.ui.canvasTabWidget.currentIndex()))
        self.currentImage = self.ui.canvasTabWidget.currentWidget().imageLabel
        self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W, self.currentImage.H))

    def tabClose(self,index):
        if self.ui.canvasTabWidget.count() == 1:
            self.saveImage()
            QApplication.exit()
        else:
            self.ui.canvasTabWidget.setCurrentIndex(index)
            self.saveImage()
            self.ui.canvasTabWidget.removeTab(index)

    # def changeEvent(self, event):
    #
    #     if event.type() == QEvent.WindowStateChange:
    #         if self.windowState() & Qt.WindowMinimized:
    #             print('changeEvent: Minimised')
    #         elif event.oldState() & Qt.WindowMinimized:
    #             print('changeEvent: Normal/Maximised/FullScreen')

    def setupTab(self, image):
        print(image.shape)
        tempWidget = tabWidget(image = image)
        index = self.filename.rfind('/')
        name = self.filename[index+1:]
        self.ui.canvasTabWidget.addTab(tempWidget, name)
        self.ui.canvasTabWidget.setCurrentIndex(len(self.ui.canvasTabWidget)-1)
        self.currentImage = self.ui.canvasTabWidget.currentWidget().imageLabel
        self.currentImage.addHistory(self.currentImage.image)
        self.currentImage.setFileName(self.filename)
        self.currentImage.setDraw(self.draw)
        color = self.currentImage.getPenColor()
        self.ui.colorPickPushButton.setStyleSheet('background-color: {};'.format(color))

    def saveImage(self,image=None):
        if image is None:
            image = self.currentImage.getBoard().toImage()

        fd, type = QFileDialog.getSaveFileName(self, "Save Image", "", "*.jpg;;*.png;;*.jpeg")
        try:
            image.save(fd)
        except:

            QMessageBox.information(self, "Error",
                                    "Unable to save image.", QMessageBox.Ok)

    def loadImage(self):
        """ This function will   load the user selected image
            and set it to label using the setPhoto function
        """

        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Image","", "JPG Files (*.jpeg *.jpg );;PNG Files (*.png)")

        if self.filename:
            self.oriImage = cv2.imread(self.filename)
            self.image = self.oriImage.copy()
            self.tmpImage = self.oriImage.copy()
            image = cv2.imread(self.filename)
            self.setupTab(image)
        else:
            QMessageBox.information(self, "Error",
                                    "Unable to open image.", QMessageBox.Ok)
            app = QApplication.instance()
            app.quit()

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
        self.currentImage.emptyRedo()
        print(self.currentImage.tmpImage.shape)
    def changeDraw(self,draw='brush'):
        self.currentImage.changeDraw(draw)

    def colorPick(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.currentImage.ChangePenColor(color=color.name())
            self.ui.colorPickPushButton.setStyleSheet('background-color: {};'.format(color.name()))

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

    def changeSize (self):
        if self.ui.widthLineEdit is not None and self.ui.heightLineEdit is not None:
            new_width = int(self.ui.widthLineEdit.text())
            new_height = int(self.ui.heightLineEdit.text())
            if new_width > 0 and new_height>0:
                self.currentImage.image = cv2.resize(self.currentImage.image,(new_width,new_height))
                self.currentImage.addHistory(self.currentImage.image)
                self.currentImage.setPhoto(self.currentImage.image)
                self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W, self.currentImage.H))

    # def LookupTable(self,x, y):
    #     spline = UnivariateSpline(x, y)
    #     return spline(range(256))

    def changeEffect(self,effect='Original'):
        img = self.currentImage.image.copy()
        img = EffectFunction.changeEffect(ori_img=img,effect=effect)
        # if effect == 'Original':
        #     img = self.currentImage.image.copy()
        # elif effect == 'Greyscale':
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # elif effect == 'Brightness':
        #     img = cv2.convertScaleAbs(img, beta=50)
        # elif effect == 'Darker':
        #     img = cv2.convertScaleAbs(img, beta=-50)
        # elif effect =='Sharp':
        #     kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        #     img = cv2.filter2D(img, -1, kernel)
        # elif effect == 'Sepia':
        #     img = np.array(img, dtype=np.float64)  # converting to float to prevent loss
        #     img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
        #                                                     [0.349, 0.686, 0.168],
        #                                                     [0.393, 0.769,
        #                                                      0.189]]))  # multipying image with special sepia matrix
        #     img[np.where(img > 255)] = 255  # normalizing values greater than 255 to 255
        #     img = np.array(img, dtype=np.uint8)
        #
        # elif effect == 'Foil':
        #     img = np.array(Image.fromarray(img).filter(EMBOSS))
        # elif effect == 'Median':
        #     img = cv2.medianBlur(img, 41)
        # elif effect == 'Gaussian':
        #     img = cv2.GaussianBlur(img, (41, 41), 0)
        # elif effect=='PencilColor':
        #     sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        #     img = sk_color
        # elif effect =='HDR':
        #     img = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        # elif effect =='Invert':
        #     img = cv2.bitwise_not(img)
        # elif effect =='Summer':
        #     increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        #     decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        #     blue_channel, green_channel, red_channel = cv2.split(img)
        #     red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        #     blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        #     img = cv2.merge((blue_channel, green_channel, red_channel))
        # elif effect =='Winter':
        #     increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        #     decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        #     blue_channel, green_channel, red_channel = cv2.split(img)
        #     red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        #     blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        #     img = cv2.merge((blue_channel, green_channel, red_channel))
        # print('filter ', img.shape)
        self.currentImage.tmpImage = img
        self.currentImage.setPhoto(self.currentImage.tmpImage)

    # defining a function
    def contrast_value(self,value):
        if value != '' and int(value) in range(1, 10):
            self.ui.contraseHorizontalSlider.blockSignals(True)
            self.ui.contrastLineEdit.blockSignals(True)

            self.ui.contrastLineEdit.setText(str(value))
            self.ui.contraseHorizontalSlider.setSliderPosition(int(value))

            self.ui.contraseHorizontalSlider.blockSignals(False)
            self.ui.contrastLineEdit.blockSignals(False)

            self.contrast_value_now = int(value)
            print('contrast: ', value)
            self.updateEffect()

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
            self.updateEffect()

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
            self.updateEffect()
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
        self.updateEffect()

    def dilate_value(self,value):

        if value != '' and int(value) in range(0, 101):
            # block signals
            self.ui.dilateHorizontalSlider.blockSignals(True)
            self.ui.dilateLineEdit.blockSignals(True)
            # update value
            self.ui.dilateLineEdit.setText(str(value))
            self.ui.dilateHorizontalSlider.setSliderPosition(int(value))
            # unblock signals
            self.ui.dilateHorizontalSlider.blockSignals(False)
            self.ui.dilateLineEdit.blockSignals(False)

            self.dilate_value_now = int(value)
            print('dilate: ', value)
            self.updateEffect()
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
            self.updateEffect()

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
            self.updateEffect()

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
            self.updateEffect()
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
                self.updateEffect()


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
            self.updateEffect()
    # # details filter
    # def changeDilate(self,img,value):
    #
    #     kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
    #     img = cv2.dilate(img, kernel_size,iterations=1)
    #     return img
    # def changeErode(self,img,value):
    #     kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
    #     img = cv2.erode(img, kernel_size)
    #     return img
    # def changeContrast(self,img,value):
    #     if value>=1 and value<=10:
    #         h, w, ch = img.shape
    #         img2 = np.zeros([h, w, ch], img.dtype)
    #         img = cv2.addWeighted(img, value, img2, 1 - value, 0)  # addWeighted函数说明如下
    #
    #     return img
    #
    # def changeBlur(self, img, value):
    #     """ inputs: img image and blur values.
    #         perform blur operation using opencv function,
    #         return  img.
    #     """
    #     kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
    #     img = cv2.blur(img, kernel_size)
    #     return img
    #
    # def changeThreshold(self,img,thresh,maxval):
    #
    #     ori_img,img = cv2.threshold(img, thresh, maxval,self.threshType)
    #     return img
    #
    # def changeBrightness(self, img, value):
    #     """
    #         This function will take an image (img) and the brightness
    #         value. It will perform the brightness change using OpenCv
    #         and after split, will merge the img and return it.
    #     """
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     v = np.int16(v)
    #     v+=value
    #     v= np.clip(v,0,255)
    #     v = np.uint8(v)
    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img
    # def changeHue(self, img, value):
    #     """
    #         This function will take an image (img) and the brightness
    #         value. It will perform the brightness change using OpenCv
    #         and after split, will merge the img and return it.
    #     """
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     h = np.int16(h)
    #     h+=value
    #     h = np.clip(h,0,255)
    #     h = np.uint8(h)
    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img
    #
    #
    #
    # def changeSaturation(self, img, value):
    #
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     s = np.int16(s)
    #     s+=value
    #     s = np.clip(s,0,255)
    #     s = np.uint8(s)
    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img
    def rotateFlip(self, option):
        image = self.currentImage.image
        image = CropRotateFunction.rotateFlip(image,option)

        # h, w, _ = self.currentImage.image.shape
        # temp = np.zeros((w, h, 3), np.uint8)  # null image with inverted dimensions
        # if option == 'LEFT':
        #     for i in range(0, w):
        #         temp[w - i - 1, :, :] = self.currentImage.image[:, i, :]
        # elif option == 'RIGHT':
        #     for j in range(0, h):
        #         temp[:, h - j - 1, :] = self.currentImage.image[j, :, :]

        self.currentImage.image = image
        self.currentImage.tmpImage = image
        self.currentImage.setPhoto(self.currentImage.image)
        self.currentImage.addHistory(self.currentImage.image.copy())
        self.currentImage.emptyRedo()


    # def flip(self, option):
    #     h, w, _ = self.currentImage.image.shape
    #     temp = np.zeros((h, w, 3), np.uint8)
    #     if option == 'HORIZONTAL':
    #         for i in range(0, w):
    #             temp[:, i, :] = self.currentImage.image[:, w - i - 1, :]
    #     elif option == 'VERTICAL':
    #         for j in range(0, h):
    #             temp[j, :, :] = self.currentImage.image[h - j - 1, :, :]
    #     self.currentImage.image = temp
    #     self.currentImage.tmpImage = temp
    #     self.currentImage.setPhoto(self.currentImage.image)
    #     self.currentImage.addHistory(self.currentImage.image.copy())
    #     self.currentImage.emptyRedo()


    def updateEffect(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        self.currentImage.tmpImage = FilterFunction.changeBrightness(self.currentImage.image, self.brightness_value_now)
        self.currentImage.tmpImage = FilterFunction.changeBlur(self.currentImage.tmpImage, self.blur_value_now)
        self.currentImage.tmpImage = FilterFunction.changeSaturation(self.currentImage.tmpImage, self.saturation_value_now)
        self.currentImage.tmpImage = FilterFunction.changeHue(self.currentImage.tmpImage, self.hue_value_now)
        self.currentImage.tmpImage = FilterFunction.changeDilate(self.currentImage.tmpImage, self.dilate_value_now)
        self.currentImage.tmpImage = FilterFunction.changeErode(self.currentImage.tmpImage, self.erode_value_now)
        self.currentImage.tmpImage =FilterFunction.changeContrast(self.currentImage.tmpImage, self.contrast_value_now)
        self.currentImage.tmpImage = FilterFunction.changeThreshold(self.currentImage.tmpImage, self.thresh_value_now,self.maxval_value_now,self.threshType)
        self.currentImage.setPhoto(self.currentImage.tmpImage)


    def handleCalc(self,text):

        print(self.ui.penStyleComboBox.currentIndex())

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self.applyEffect()
            self.currentImage.setPhoto(self.currentImage.image)
            self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W, self.currentImage.H))

#
# app = QApplication(sys.argv)
# stats = Stats()
# stats.show()
# app.exec_()

from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QColorDialog,QMessageBox
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, QEvent
from UI import Ui_MainWindow
import cv2
import numpy as np
from PIL import Image
from PIL.ImageFilter import EMBOSS
from scipy.interpolate import UnivariateSpline
from ImageCanvas import tabWidget
from rembg import remove
import tensorflow as tf
from Functions.AdvancedFunction import pixelate,kMeansImage,augContrast
from InProgress import ProgressBar
import time
from Functions import  EffectFunction,FilterFunction,CropRotateFunction


class Stats(QMainWindow):
    def __init__(self):

        super().__init__()
        self.historyImage = []
        self.draw =False
        self.crop =False
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
        self.contrast_value_now = 0
        self.threshType = cv2.THRESH_TOZERO
        self.ui = Ui_MainWindow()
        # 初始化界面
        self.ui.setupUi(self)
        self.loadImage()
        self.setupSign()
        self.showMaximized()

    def setupSign(self):
        # # 信号和槽
        self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W,self.currentImage.H))
        self.ui.comparePushButton.pressed.connect(self.currentImage.comparePressedImage)
        self.ui.comparePushButton.released.connect(self.currentImage.compareReleasedImage)
        self.ui.actionsave.triggered.connect(self.saveImage)
        self.ui.stackedWidget.currentChanged.connect(self.changeStacked)
        self.ui.canvasTabWidget.tabCloseRequested.connect(self.tabClose)
        self.ui.canvasTabWidget.currentChanged.connect(self.tabChange)
        self.ui.actionNew.triggered.connect(self.loadImage)
        self.ui.actionUndo.triggered.connect(self.undoImage)
        self.ui.actionRedo.triggered.connect(self.redoImage)
        self.ui.actionGet_Start.triggered.connect(self.help)
        self.ui.actionOriginalImage.triggered.connect(self.setOriginalImage)
        # self.ui.cropButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(5))
        self.ui.rotateButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.drawButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.filterButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.effectButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.advancedButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(4))
        # filter function
        self.ui.resetALLPushButton.clicked.connect(self.resetAll)
        self.ui.brightnessHorizontalSlider.valueChanged['int'].connect(self.brightness_value)
        self.ui.blurHorizontalSlider.valueChanged['int'].connect(self.blur_value)
        self.ui.saturationHorizontalSlider.valueChanged['int'].connect(self.saturation_value)
        self.ui.hueHorizontalSlider.valueChanged['int'].connect(self.hue_value)
        self.ui.dilateHorizontalSlider.valueChanged['int'].connect(self.dilate_value)
        self.ui.erodeHorizontalSlider.valueChanged['int'].connect(self.erode_value)
        self.ui.threshHorizontalSlider.valueChanged['int'].connect(self.thresh_value)
        self.ui.maxvalHorizontalSlider.valueChanged['int'].connect(self.maxval_value)
        self.ui.contraseHorizontalSlider.valueChanged['int'].connect(self.contrast_value)

        self.ui.contrastLineEdit.setValidator(QIntValidator(1, 10, self))
        self.ui.contrastLineEdit.textChanged.connect(self.contrast_value)

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


        # crop function
        self.ui.widthLineEdit.setValidator(QIntValidator(1, 5000, self))
        self.ui.heightLineEdit.setValidator(QIntValidator(1, 5000, self))
        self.ui.widthLineEdit.textChanged.connect(self.fixRatioWidth)
        self.ui.heightLineEdit.textChanged.connect(self.fixRatioHeight)
        self.ui.sizePushButton.clicked.connect(self.changeSize)
        self.ui.rotateLeft.clicked.connect(lambda: self.rotateFlip('LEFT'))
        self.ui.rotateRight.clicked.connect(lambda: self.rotateFlip('RIGHT'))
        self.ui.horizontalPushButton.clicked.connect(lambda: self.rotateFlip('HORIZONTAL'))
        self.ui.verticalPushButton.clicked.connect(lambda: self.rotateFlip('VERTICAL'))
        self.ui.cropPushButton.clicked.connect(self.changeCrop)

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
        # self.ui.effectApplyButton.clicked.connect(self.applyEffect)

        # advanced part
        self.ui.backRemovePushButton.clicked.connect(self.removeBackground)
        # self.ui.transferPushButton.clicked.connect(self.transferStyle)
        self.ui.autoContrastPushButton.clicked.connect(self.autoContrast)
        self.ui.pixel32PushButton.clicked.connect(lambda: self.changeToPiexl(size=32))
        self.ui.pixel48PushButton.clicked.connect(lambda: self.changeToPiexl(size=48))
        self.ui.pixel64PushButton.clicked.connect(lambda: self.changeToPiexl(size=64))
        self.ui.pixel128PushButton.clicked.connect(lambda: self.changeToPiexl(size=128))

    def resetAll(self):
        for i in self.ui.filterPage.findChildren(QtWidgets.QSlider):
            i.blockSignals(True)
            i.setValue(0)
        for i in self.ui.filterPage.findChildren(QtWidgets.QLineEdit):
            i.blockSignals(True)
            i.setText('0')
        self.ui.contrastLineEdit.setText('1')
        self.ui.contraseHorizontalSlider.setValue(1)

        self.brightness_value_now = 0
        self.blur_value_now = 0
        self.saturation_value_now = 0
        self.hue_value_now = 0
        self.dilate_value_now = 0
        self.erode_value_now = 0
        self.thresh_value_now = 0
        self.maxval_value_now = 0
        self.contrast_value_now = 1

        for i in self.ui.filterPage.findChildren(QtWidgets.QSlider):
            i.blockSignals(False)

        for i in self.ui.filterPage.findChildren(QtWidgets.QLineEdit):
            i.blockSignals(False)

        self.threholdReset()
        self.updateEffect()




    def help(self):
        import os
        os.startfile('Help.pdf')
        # return
    def fixRatioWidth(self,value):
        if self.ui.fixScaleCheckBox.isChecked():
            if value != '':
                width = int(value)
                height = int(width * self.currentImage.getRatio())
                # print(height)
                self.ui.heightLineEdit.blockSignals(True)
                self.ui.heightLineEdit.setText(str(height))
                self.ui.heightLineEdit.blockSignals(False)
    def fixRatioHeight(self,value):
        if self.ui.fixScaleCheckBox.isChecked():
            if value != '' :
                # print(1111)
                height = int(value)
                width = int(height/self.currentImage.getRatio())
                # print(width)
                self.ui.widthLineEdit.blockSignals(True)
                self.ui.widthLineEdit.setText(str(width))
                self.ui.widthLineEdit.blockSignals(False)
    def autoContrast(self):
        img = self.currentImage.image.copy()
        aug_img = augContrast(img)
        self.currentImage.tmpImage = aug_img
        self.currentImage.setPhoto(self.currentImage.tmpImage)

    def changeToPiexl(self,size):

        bar = ProgressBar(self)

        bar.set_value(0)
        img = self.currentImage.image.copy()
        time.sleep(0.1)
        bar.set_value(10)
        imgPiexl = pixelate(img, size, size)
        time.sleep(0.1)
        bar.set_value(50)
        newImage = kMeansImage(imgPiexl, 5)
        time.sleep(0.1)
        bar.set_value(80)
        self.currentImage.tmpImage = newImage
        self.currentImage.setPhoto(self.currentImage.tmpImage)
        time.sleep(0.1)
        bar.set_value(100)
        time.sleep(0.1)
        bar.close()
        QApplication.processEvents()


    def changeCrop(self):
        if self.crop == True:
            self.crop = False
            self.setCropButtonColor()

        else:
            self.crop = True
            self.setCropButtonColor()


        self.currentImage.setCrop(self.crop)
    def setCropButtonColor(self):
        if not self.crop:
            self.ui.cropPushButton.setStyleSheet("""
                        QPushButton{
                            border:2px solid rgb(255, 255, 255);
                            background-color:rgb(0, 0, 0);
                            color: rgb(255, 255, 255)
                        }
                        QPushButton:hover{

                            border: 2px solid rgb(0, 154, 206)
                        }
                        """)
        else:
            self.ui.cropPushButton.setStyleSheet("""
                                    QPushButton{
                                        border:2px solid rgb(0, 0, 0);
                                        background-color:rgb(255, 255, 255);
                                        color: rgb(0, 0, 0)
                                    }
                                    QPushButton:hover{

                                        border: 2px solid rgb(0, 154, 206)
                                    }
                                    """)


    def changeStacked(self):

        if self.ui.stackedWidget.currentIndex() == 1:
            self.draw = True
        else:
            self.draw = False
            self.crop = False
            self.setCropButtonColor()


        self.currentImage.setDraw(self.draw)
        self.currentImage.setCrop(self.crop)

    def tensor_to_image(self,tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        # img = PIL.Image.fromarray(tensor)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return np.squeeze(tensor, axis=0)

    def tf_load_img(self,path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def transferStyle(self):
        pass
        # filename, _ = QFileDialog.getOpenFileName(self, "Open Transfer Image", "",
        #                                                "JPG Files (*.jpeg *.jpg );;PNG Files (*.png)")
        # if filename:
        #     content_image = self.tf_load_img(self.filename)
        #     style_image =self.tf_load_img(filename)
        #     stylized_image = self.hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        #     # print(stylized_image.shape)
        #     # np.float32(stylized_image)
        #     self.currentImage.setPhoto(np.float32(stylized_image))


    def removeBackground(self):
        bar = ProgressBar(self)
        bar.set_value(0)
        time.sleep(0.1)
        img = self.currentImage.image.copy()
        bar.set_value(20)
        result = remove(img)
        time.sleep(0.1)
        bar.set_value(50)
        new_result = cv2.cvtColor(result,cv2.COLOR_BGRA2BGR)
        time.sleep(0.1)
        bar.set_value(80)
        self.currentImage.tmpImage = new_result
        self.currentImage.setPhoto(self.currentImage.tmpImage)
        time.sleep(0.1)
        bar.set_value(100)
        time.sleep(0.1)
        bar.close()

    def redoImage(self):
        self.currentImage.redo()

    def undoImage(self):
        self.currentImage.undo()



    def tabChange(self):

        # print('change to {} tab'.format(self.ui.canvasTabWidget.currentIndex()))
        self.currentImage = self.ui.canvasTabWidget.currentWidget().imageLabel
        self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W, self.currentImage.H))

    def tabClose(self,index):
        if self.ui.canvasTabWidget.count() == 1:
            self.saveImage()
            QApplication.exit()
        else:
            self.ui.canvasTabWidget.setCurrentIndex(index)
            self.saveImage()
            self.ui.canvasTabWidget.removeTab(index)

    # def changeEvent(self, event):
    #
    #     if event.type() == QEvent.WindowStateChange:
    #         if self.windowState() & Qt.WindowMinimized:
    #             print('changeEvent: Minimised')
    #         elif event.oldState() & Qt.WindowMinimized:
    #             print('changeEvent: Normal/Maximised/FullScreen')

    def setupTab(self, image):
        print(image.shape)
        tempWidget = tabWidget(image = image)
        index = self.filename.rfind('/')
        name = self.filename[index+1:]
        self.ui.canvasTabWidget.addTab(tempWidget, name)
        self.ui.canvasTabWidget.setCurrentIndex(len(self.ui.canvasTabWidget)-1)
        self.currentImage = self.ui.canvasTabWidget.currentWidget().imageLabel
        self.currentImage.addHistory(self.currentImage.image)
        self.currentImage.setFileName(self.filename)
        self.currentImage.setDraw(self.draw)
        color = self.currentImage.getPenColor()
        self.ui.colorPickPushButton.setStyleSheet('background-color: {};'.format(color))

    def saveImage(self,image=None):
        if image is None:
            image = self.currentImage.getBoard().toImage()

        fd, type = QFileDialog.getSaveFileName(self, "Save Image", "", "*.jpg;;*.png;;*.jpeg")
        try:
            image.save(fd)
        except:

            QMessageBox.information(self, "Error",
                                    "Unable to save image.", QMessageBox.Ok)

    def loadImage(self):
        """ This function will   load the user selected image
            and set it to label using the setPhoto function
        """

        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Image","", "JPG Files (*.jpeg *.jpg );;PNG Files (*.png)")

        if self.filename:
            self.oriImage = cv2.imread(self.filename)
            self.image = self.oriImage.copy()
            self.tmpImage = self.oriImage.copy()
            image = cv2.imread(self.filename)
            self.setupTab(image)
        else:
            QMessageBox.information(self, "Error",
                                    "Unable to open image.", QMessageBox.Ok)
            app = QApplication.instance()
            app.quit()

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
        self.currentImage.emptyRedo()
        print(self.currentImage.tmpImage.shape)
    def changeDraw(self,draw='brush'):
        self.currentImage.changeDraw(draw)

    def colorPick(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.currentImage.ChangePenColor(color=color.name())
            self.ui.colorPickPushButton.setStyleSheet('background-color: {};'.format(color.name()))

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

    def changeSize (self):
        if self.ui.widthLineEdit is not None and self.ui.heightLineEdit is not None:
            new_width = int(self.ui.widthLineEdit.text())
            new_height = int(self.ui.heightLineEdit.text())
            if new_width > 0 and new_height>0:
                self.currentImage.image = cv2.resize(self.currentImage.image,(new_width,new_height))
                self.currentImage.addHistory(self.currentImage.image)
                self.currentImage.setPhoto(self.currentImage.image)
                self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W, self.currentImage.H))

    # def LookupTable(self,x, y):
    #     spline = UnivariateSpline(x, y)
    #     return spline(range(256))

    def changeEffect(self,effect='Original'):
        img = self.currentImage.image.copy()
        img = EffectFunction.changeEffect(ori_img=img,effect=effect)
        # if effect == 'Original':
        #     img = self.currentImage.image.copy()
        # elif effect == 'Greyscale':
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # elif effect == 'Brightness':
        #     img = cv2.convertScaleAbs(img, beta=50)
        # elif effect == 'Darker':
        #     img = cv2.convertScaleAbs(img, beta=-50)
        # elif effect =='Sharp':
        #     kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        #     img = cv2.filter2D(img, -1, kernel)
        # elif effect == 'Sepia':
        #     img = np.array(img, dtype=np.float64)  # converting to float to prevent loss
        #     img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
        #                                                     [0.349, 0.686, 0.168],
        #                                                     [0.393, 0.769,
        #                                                      0.189]]))  # multipying image with special sepia matrix
        #     img[np.where(img > 255)] = 255  # normalizing values greater than 255 to 255
        #     img = np.array(img, dtype=np.uint8)
        #
        # elif effect == 'Foil':
        #     img = np.array(Image.fromarray(img).filter(EMBOSS))
        # elif effect == 'Median':
        #     img = cv2.medianBlur(img, 41)
        # elif effect == 'Gaussian':
        #     img = cv2.GaussianBlur(img, (41, 41), 0)
        # elif effect=='PencilColor':
        #     sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        #     img = sk_color
        # elif effect =='HDR':
        #     img = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        # elif effect =='Invert':
        #     img = cv2.bitwise_not(img)
        # elif effect =='Summer':
        #     increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        #     decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        #     blue_channel, green_channel, red_channel = cv2.split(img)
        #     red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        #     blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        #     img = cv2.merge((blue_channel, green_channel, red_channel))
        # elif effect =='Winter':
        #     increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        #     decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        #     blue_channel, green_channel, red_channel = cv2.split(img)
        #     red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        #     blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        #     img = cv2.merge((blue_channel, green_channel, red_channel))
        # print('filter ', img.shape)
        self.currentImage.tmpImage = img
        self.currentImage.setPhoto(self.currentImage.tmpImage)

    # defining a function
    def contrast_value(self,value):
        if value != '' and int(value) in range(1, 10):
            self.ui.contraseHorizontalSlider.blockSignals(True)
            self.ui.contrastLineEdit.blockSignals(True)

            self.ui.contrastLineEdit.setText(str(value))
            self.ui.contraseHorizontalSlider.setSliderPosition(int(value))

            self.ui.contraseHorizontalSlider.blockSignals(False)
            self.ui.contrastLineEdit.blockSignals(False)

            self.contrast_value_now = int(value)
            print('contrast: ', value)
            self.updateEffect()

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
            self.updateEffect()

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
            self.updateEffect()
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
        self.updateEffect()

    def dilate_value(self,value):

        if value != '' and int(value) in range(0, 101):
            # block signals
            self.ui.dilateHorizontalSlider.blockSignals(True)
            self.ui.dilateLineEdit.blockSignals(True)
            # update value
            self.ui.dilateLineEdit.setText(str(value))
            self.ui.dilateHorizontalSlider.setSliderPosition(int(value))
            # unblock signals
            self.ui.dilateHorizontalSlider.blockSignals(False)
            self.ui.dilateLineEdit.blockSignals(False)

            self.dilate_value_now = int(value)
            print('dilate: ', value)
            self.updateEffect()
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
            self.updateEffect()

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
            self.updateEffect()

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
            self.updateEffect()
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
                self.updateEffect()


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
            self.updateEffect()
    # # details filter
    # def changeDilate(self,img,value):
    #
    #     kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
    #     img = cv2.dilate(img, kernel_size,iterations=1)
    #     return img
    # def changeErode(self,img,value):
    #     kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
    #     img = cv2.erode(img, kernel_size)
    #     return img
    # def changeContrast(self,img,value):
    #     if value>=1 and value<=10:
    #         h, w, ch = img.shape
    #         img2 = np.zeros([h, w, ch], img.dtype)
    #         img = cv2.addWeighted(img, value, img2, 1 - value, 0)  # addWeighted函数说明如下
    #
    #     return img
    #
    # def changeBlur(self, img, value):
    #     """ inputs: img image and blur values.
    #         perform blur operation using opencv function,
    #         return  img.
    #     """
    #     kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
    #     img = cv2.blur(img, kernel_size)
    #     return img
    #
    # def changeThreshold(self,img,thresh,maxval):
    #
    #     ori_img,img = cv2.threshold(img, thresh, maxval,self.threshType)
    #     return img
    #
    # def changeBrightness(self, img, value):
    #     """
    #         This function will take an image (img) and the brightness
    #         value. It will perform the brightness change using OpenCv
    #         and after split, will merge the img and return it.
    #     """
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     v = np.int16(v)
    #     v+=value
    #     v= np.clip(v,0,255)
    #     v = np.uint8(v)
    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img
    # def changeHue(self, img, value):
    #     """
    #         This function will take an image (img) and the brightness
    #         value. It will perform the brightness change using OpenCv
    #         and after split, will merge the img and return it.
    #     """
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     h = np.int16(h)
    #     h+=value
    #     h = np.clip(h,0,255)
    #     h = np.uint8(h)
    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img
    #
    #
    #
    # def changeSaturation(self, img, value):
    #
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     s = np.int16(s)
    #     s+=value
    #     s = np.clip(s,0,255)
    #     s = np.uint8(s)
    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img
    def rotateFlip(self, option):
        image = self.currentImage.image
        image = CropRotateFunction.rotateFlip(image,option)

        # h, w, _ = self.currentImage.image.shape
        # temp = np.zeros((w, h, 3), np.uint8)  # null image with inverted dimensions
        # if option == 'LEFT':
        #     for i in range(0, w):
        #         temp[w - i - 1, :, :] = self.currentImage.image[:, i, :]
        # elif option == 'RIGHT':
        #     for j in range(0, h):
        #         temp[:, h - j - 1, :] = self.currentImage.image[j, :, :]

        self.currentImage.image = image
        self.currentImage.tmpImage = image
        self.currentImage.setPhoto(self.currentImage.image)
        self.currentImage.addHistory(self.currentImage.image.copy())
        self.currentImage.emptyRedo()


    # def flip(self, option):
    #     h, w, _ = self.currentImage.image.shape
    #     temp = np.zeros((h, w, 3), np.uint8)
    #     if option == 'HORIZONTAL':
    #         for i in range(0, w):
    #             temp[:, i, :] = self.currentImage.image[:, w - i - 1, :]
    #     elif option == 'VERTICAL':
    #         for j in range(0, h):
    #             temp[j, :, :] = self.currentImage.image[h - j - 1, :, :]
    #     self.currentImage.image = temp
    #     self.currentImage.tmpImage = temp
    #     self.currentImage.setPhoto(self.currentImage.image)
    #     self.currentImage.addHistory(self.currentImage.image.copy())
    #     self.currentImage.emptyRedo()


    def updateEffect(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        self.currentImage.tmpImage = FilterFunction.changeBrightness(self.currentImage.image, self.brightness_value_now)
        self.currentImage.tmpImage = FilterFunction.changeBlur(self.currentImage.tmpImage, self.blur_value_now)
        self.currentImage.tmpImage = FilterFunction.changeSaturation(self.currentImage.tmpImage, self.saturation_value_now)
        self.currentImage.tmpImage = FilterFunction.changeHue(self.currentImage.tmpImage, self.hue_value_now)
        self.currentImage.tmpImage = FilterFunction.changeDilate(self.currentImage.tmpImage, self.dilate_value_now)
        self.currentImage.tmpImage = FilterFunction.changeErode(self.currentImage.tmpImage, self.erode_value_now)
        self.currentImage.tmpImage =FilterFunction.changeContrast(self.currentImage.tmpImage, self.contrast_value_now)
        self.currentImage.tmpImage = FilterFunction.changeThreshold(self.currentImage.tmpImage, self.thresh_value_now,self.maxval_value_now,self.threshType)
        self.currentImage.setPhoto(self.currentImage.tmpImage)


    def handleCalc(self,text):

        print(self.ui.penStyleComboBox.currentIndex())

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self.applyEffect()
            self.currentImage.setPhoto(self.currentImage.image)
            self.ui.statusbar.showMessage("Image Size : {} X {}".format(self.currentImage.W, self.currentImage.H))

#
# app = QApplication(sys.argv)
# stats = Stats()
# stats.show()
# app.exec_()

