import numpy as np
import math
import cv2

def brightness(image, scale=100):
    image = image.astype(np.uint16)
    image = np.clip(image + scale, 0, 255).astype(np.uint8)
    return image

def contrast(image, factor=0.5):
    image = np.clip((image.astype(np.float32) * factor), 0, 255).astype(np.uint8)
    return image

def contrast_stretch(image):
    minV, maxV = np.min(image), np.max(image)
    stretched = ((image - minV) / (maxV - minV) * 255).astype(np.uint8)
    return stretched

def negative(image):
    neg = 255 - image
    return neg

def negative_gray(image):
    return 255 - image

def binarization(image, threshold=180):
    binarized = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binarized