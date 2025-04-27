import numpy as np
import cv2

def histogram_equalization(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    return cv2.equalizeHist(image_gray)

def histogram_rgb(image):
    channels = cv2.split(image)
    hist_images = []
    for ch in channels:
        hist_images.append(cv2.equalizeHist(ch))
    return cv2.merge(hist_images)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def adder(image, image2):
    # Samakan ukuran
    if image.shape != image2.shape:
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

    # Samakan tipe data
    if image.dtype != image2.dtype:
        image2 = image2.astype(image.dtype)

    image2_resized = cv2.resize(image2, (image.shape[1], image.shape[0]))
    result = cv2.add(image, image2_resized)
    return result

def subs(image, image2):
    # Samakan ukuran
    if image.shape != image2.shape:
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

    # Samakan tipe data
    if image.dtype != image2.dtype:
        image2 = image2.astype(image.dtype)

    image2_resized = cv2.resize(image2, (image.shape[1], image.shape[0]))
    result = cv2.subtract(image, image2_resized)
    return result

def logic_and(image, image2):
    # Samakan ukuran
    if image.shape != image2.shape:
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

    # Samakan tipe data
    if image.dtype != image2.dtype:
        image2 = image2.astype(image.dtype)

    return cv2.bitwise_and(image, image2)

def logic_or(image, image2):
    # Samakan ukuran
    if image.shape != image2.shape:
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

    # Samakan tipe data
    if image.dtype != image2.dtype:
        image2 = image2.astype(image.dtype)

    return cv2.bitwise_or(image, image2)

def logic_xor(image, image2):
    # Samakan ukuran
    if image.shape != image2.shape:
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

    # Samakan tipe data
    if image.dtype != image2.dtype:
        image2 = image2.astype(image.dtype)

    return cv2.bitwise_xor(image, image2)