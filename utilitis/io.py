import cv2
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def loadimg(self):
    filename, _ = QFileDialog.getOpenFileName(
        self,
        "Load Image",
        "",
        "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)"
    )
    if filename:
        image = cv2.imread(filename)
        if image is not None:
            return image
        else:
            QMessageBox.warning(self, "Error", "Failed to Load Image.")
    return None

def saveimg(self, image):
    if image is None:
        QMessageBox.warning(self, "Error", "Tidak ada gambar untuk disimpan.")
        return

    filename, _ = QFileDialog.getSaveFileName(
        self,
        "Save File",
        "",
        "JPEG Files (*.jpg);;PNG Files (*.png);;Bitmap Files (*.bmp);;All Files (*)"
    )

    if filename:
        success = cv2.imwrite(filename, image)
        if success:
            QMessageBox.information(self, "Success", "Gambar berhasil disimpan.")
        else:
            QMessageBox.warning(self, "Error", "Gagal menyimpan gambar.")