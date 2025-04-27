import sys

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

from utilitis.io import loadimg, saveimg, array2qimage
from utilitis.grayscale import grayscaling

# Taken from "materi" lib
from materi.A1A8 import brightness, contrast, contrast_stretch, negative, negative_gray, binarization # Buat materi di A1-A8
from materi.A9C2 import histogram_equalization, histogram_rgb, rotate_image, adder, subs, logic_and, \
    logic_or, logic_xor # Buat materi A9-C2
from materi.D1D6 import convolve2d, convolve_kernel1, convolve_kernel2, mean, gaussian, median, max, sharpening_kernel1, \
    sharpening_kernel2, sharpening_kernel3, sharpening_kernel4, sharpening_kernel5, sharpening_kernel6, \
    sharpening_kernel7  # Buat materi di D1-D6
from materi.E1E2 import fourier_transform, dft2d # Buat materi di E1 & E2
from materi.F1F2 import sobel, prewitt, roberts, laplacian, log, canny # Buat materi di F1 & F2

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.centerWindow()
        self.Image = None
        self.Image2 = None
        self.Image3 = None

        # BASE FUNCTION MENU
        self.button_loadimg1.clicked.connect(self.loadF1)
        self.button_loadimg2.clicked.connect(self.loadF2)
        self.button_saveimg.clicked.connect(self.saveF)
        self.button_grayscale.clicked.connect(self.gs)
        self.button_clnW1.clicked.connect(self.clnw1)
        self.button_clnW2.clicked.connect(self.clnw2)
        self.button_clnW3.clicked.connect(self.clnw3)

        # BASE OPERATION MENU
        self.actionBrightAdj.triggered.connect(self.ba)
        self.actionCtrstAdj.triggered.connect(self.ca)
        self.actionCtrst_StrcAdj.triggered.connect(self.cs)
        self.actionNgte.triggered.connect(self.nv)
        self.actionNgte_G.triggered.connect(self.ng)
        self.actionBin.triggered.connect(self.bin)

        # HISTOGRAM OPERTATION MENU
        self.actiongG_Hstgrm.triggered.connect(self.gh)
        self.actionRGB_Hstgrm.triggered.connect(self.rgbh)
        self.actionEqualizer.triggered.connect(self.eqh)

        # GEOMETRIC HISTOGRAM MENU
        # self.actionTranslation.triggered.connect(self.tr)
        self.actionRotation.triggered.connect(self.rt)
        # self.actionResizing.triggered.connect(self.rs)
        # self.actionCropping.triggered.connect(self.cp)

        # LOGIC OPERATION MENU
        self.actionAdd.triggered.connect(self.add)
        self.actionSubs.triggered.connect(self.subs)
        self.actionAnd.triggered.connect(self.And)
        self.actionOr.triggered.connect(self.Or)
        self.actionXor.triggered.connect(self.Xor)

        # CONVOLE & FILTER OPERATION MENU
        self.actionConvKn1.triggered.connect(self.convkn1)
        self.actionConvKn2.triggered.connect(self.convkn2)
        self.actionMean2x2.triggered.connect(self.mn2x2)
        self.actionMean3x3.triggered.connect(self.mn3x3)
        self.actionGaussian.triggered.connect(self.gaus)
        self.actionMedian.triggered.connect(self.mdn)
        self.actionSharpKn1.triggered.connect(self.sharpkn1)
        self.actionSharpKn2.triggered.connect(self.sharpkn2)
        self.actionSharpKn3.triggered.connect(self.sharpkn3)
        self.actionSharpKn4.triggered.connect(self.sharpkn4)
        self.actionSharpKn5.triggered.connect(self.sharpkn5)
        self.actionSharpKn6.triggered.connect(self.sharpkn6)
        self.actionSharpKn7.triggered.connect(self.sharpkn7)
        self.actionMax.triggered.connect(self.max)

        # FOURIER TRANSFORMATION MENU
        self.actionFT.triggered.connect(self.ft)
        self.actionDFT.triggered.connect(self.dft)

        # EDGE DETECTION MENU
        self.actionSbl.triggered.connect(self.sbl)
        self.actionPrwt.triggered.connect(self.prwt)
        self.actionRbts.triggered.connect(self.rbts)
        self.actionLplc.triggered.connect(self.lplc)
        self.actionLoG.triggered.connect(self.log)
        self.actionCanny.triggered.connect(self.cny)

    def centerWindow(self):
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry().center()
        frameGm.moveCenter(screen)
        self.move(frameGm.topLeft())

    # BASE FUNCTION'S FUNCTION
    def loadF1(self):
        image = loadimg(self)
        if image is not None:
            self.Image = image
            self.displayImage(1)

    def loadF2(self):
        image = loadimg(self)
        if image is not None:
            self.Image2 = image
            self.displayImage(2)

    def saveF(self):
        saveimg(self, self.Image3)

    def gs(self):
        if self.Image is not None:
            try:
                self.Image = grayscaling(self.Image)
            except:
                pass
            self.displayImage(1)
        if self.Image2 is not None:
            try:
                self.Image2 = grayscaling(self.Image2)
            except:
                pass
            self.displayImage(2)

    def clnw1(self):
        self.displayImage(1, True)
        self.Image = None

    def clnw2(self):
        self.displayImage(2, True)
        self.Image2 = None

    def clnw3(self):
        self.displayImage(3, True)
        self.Image3 = None



    # BASIC OPERATION FUNCTION
    def ba(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = brightness(self.Image, 100)
        self.Image3 = hasil
        self.displayImage(3)

    def ca(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = contrast(self.Image, 1.5)
        self.Image3 = hasil
        self.displayImage(3)

    def cs(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = contrast_stretch(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def nv(self):
        hasil = negative(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def ng(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = negative_gray(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def bin(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = binarization(self.Image, 150)
        self.Image3 = hasil
        self.displayImage(3)



    # HISTOGRAM OPERATION FUNCTION
    def gh(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = histogram_rgb(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def rgbh(self):
        hasil = histogram_rgb(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def eqh(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = histogram_equalization(self.Image)
        self.Image3 = hasil
        self.displayImage(3)



    # GEOMETRIC OPERATION FUNCTION
    def rt(self):
        hasil = rotate_image(self.Image, 90)
        self.Image3 = hasil
        self.displayImage(3)




    # LOGIC OPERATION FUNCTION
    def add(self):
        try:
            self.Image = grayscaling(self.Image)
            self.Image2 = grayscaling(self.Image2)
        except:
            pass

        hasil = adder(self.Image, self.Image2)
        self.Image3 = hasil
        self.displayImage(3)

    def subs(self):
        try:
            self.Image = grayscaling(self.Image)
            self.Image2 = grayscaling(self.Image2)
        except:
            pass

        hasil = subs(self.Image, self.Image2)
        self.Image3 = hasil
        self.displayImage(3)

    def And(self):
        try:
            self.Image = grayscaling(self.Image)
            self.Image2 = grayscaling(self.Image2)
        except:
            pass

        hasil = logic_and(self.Image, self.Image2)
        self.Image3 = hasil
        self.displayImage(3)

    def Or(self):
        try:
            self.Image = grayscaling(self.Image)
            self.Image2 = grayscaling(self.Image2)
        except:
            pass

        hasil = logic_or(self.Image, self.Image2)
        self.Image3 = hasil
        self.displayImage(3)

    def Xor(self):
        try:
            self.Image = grayscaling(self.Image)
            self.Image2 = grayscaling(self.Image2)
        except:
            pass

        hasil = logic_xor(self.Image, self.Image2)
        self.Image3 = hasil
        self.displayImage(3)



    # CONVOLUTION & FILTERATION FUNCTION
    def convkn1(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, convolve_kernel1())
        self.Image3 = hasil
        self.displayImage(3)

    def convkn2(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, convolve_kernel2())
        self.Image3 = hasil
        self.displayImage(3)

    def mn2x2(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = mean(self.Image, 2)
        self.Image3 = hasil
        self.displayImage(3)

    def mn3x3(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = mean(self.Image, 3)
        self.Image3 = hasil
        self.displayImage(3)

    def gaus(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = gaussian(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def mdn(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = median(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn1(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, sharpening_kernel1())
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn2(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, sharpening_kernel2())
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn3(self):
        # try:
        #     self.Image = grayscaling(self.Image)
        # except:
        #     pass

        hasil = convolve2d(self.Image, sharpening_kernel3())
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn4(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, sharpening_kernel4())
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn5(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, sharpening_kernel5())
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn6(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, sharpening_kernel6())
        self.Image3 = hasil
        self.displayImage(3)

    def sharpkn7(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = convolve2d(self.Image, sharpening_kernel7())
        self.Image3 = hasil
        self.displayImage(3)

    def max(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = max(self.Image)
        self.Image3 = hasil
        self.displayImage(3)



    # FOURIER TRANSFORM FUNCTION
    def ft(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = fourier_transform(self.Image)
        hasil = array2qimage(hasil)
        self.Image3 = hasil
        self.displayImage(3)

    def dft(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = dft2d(self.Image)
        hasil = array2qimage(hasil)
        self.Image3 = hasil
        self.displayImage(3)



    # EDGE DETECTION FUNCTION
    def sbl(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = sobel(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def prwt(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = prewitt(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def rbts(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = roberts(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def lplc(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = laplacian(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def log(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = log(self.Image)
        self.Image3 = hasil
        self.displayImage(3)

    def cny(self):
        try:
            self.Image = grayscaling(self.Image)
        except:
            pass

        hasil = canny(self.Image, 3, 1.4, low_threshold=50, high_threshold=150)
        self.Image3 = hasil
        self.displayImage(3)

    def displayImage(self, windows=1, clear=False):
        qformat = QImage.Format_Indexed8

        if clear:
            if windows == 1:
                self.label_window1.clear()
            elif windows == 2:
                self.label_window2.clear()
            elif windows == 3:
                self.label_window3.clear()
            return

        # Pilih image sesuai window
        if windows == 1:
            img_data = self.Image
        elif windows == 2:
            img_data = self.Image2
        elif windows == 3:
            img_data = self.Image3
        else:
            return

        if img_data is None:
            return

        # Pastikan img_data adalah numpy.ndarray sebelum memeriksa shape
        if isinstance(img_data, np.ndarray):
            if len(img_data.shape) == 3:
                if img_data.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
        elif isinstance(img_data, QImage):
            # Kalau img_data sudah QImage, langsung konversi formatnya
            qformat = QImage.Format_RGB888

        # Jika img_data adalah numpy.ndarray, konversi ke QImage
        if isinstance(img_data, np.ndarray):
            img = QImage(img_data.data, img_data.shape[1], img_data.shape[0], img_data.strides[0], qformat)
            img = img.rgbSwapped()
        else:
            img = img_data  # Jika sudah QImage, tidak perlu konversi lagi

        pixmap = QPixmap.fromImage(img)
        scaled_pixmap1 = pixmap.scaled(self.label_window1.size(), QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
        scaled_pixmap2 = pixmap.scaled(self.label_window2.size(), QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
        scaled_pixmap3 = pixmap.scaled(self.label_window3.size(), QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)

        if windows == 1:
            self.label_window1.setPixmap(scaled_pixmap1)
            self.label_window1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        elif windows == 2:
            self.label_window2.setPixmap(scaled_pixmap2)
            self.label_window2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        elif windows == 3:
            self.label_window3.setPixmap(scaled_pixmap3)
            self.label_window3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('DIGITAL IMAGE PROCESSING')
window.show()
sys.exit(app.exec())