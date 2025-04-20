import numpy as np
import cv2

# Fungsi dasar konvolusi
def convimg(image, kernel):
    Hk, Wk = kernel.shape
    pad_h = Hk // 2
    pad_w = Wk // 2

    # Pad gambar agar kernel bisa diterapkan di seluruh area
    padded_img = cv2.copyMakeBorder(image, pad_h, pad_h + (Hk % 2 == 0), pad_w, pad_w + (Wk % 2 == 0), cv2.BORDER_REFLECT)
    HImg, WImg = image.shape[:2]
    convout = np.zeros((HImg, WImg), dtype=np.float32)

    for i in range(HImg):
        for j in range(WImg):
            region = padded_img[i:i+Hk, j:j+Wk]
            convout[i, j] = np.sum(region * kernel)

    return np.clip(convout, 0, 255).astype(np.uint8)


# Convolve Kernel Options
def convolve_kernel1():
    return np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]], dtype=np.float32)

def convolve_kernel2():
    return np.array([[6,  0, -6],
                     [6,  1, -6],
                     [6,  0, -6]], dtype=np.float32)

# Convolve Kernel Options
def sharpening_kernel1():
    return np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]], dtype=np.float32)

def mean_2x2():
    return 2

def mean_3x3():
    return 3

def sharpening_kernel2():
    return np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]], dtype=np.float32)

def sharpening_kernel3():
    return np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]], dtype=np.float32)

def sharpening_kernel4():
    return np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype=np.float32)

def sharpening_kernel5():
    return np.array([[1, -2, 1],
                    [-2, 5, -2],
                    [1, -2, 1]], dtype=np.float32)

def sharpening_kernel6():
    return np.array([[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]], dtype=np.float32)

def sharpening_kernel7():
    return np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)

# Convolve2d
def convolve2d(image, kernel):
    kernel = kernel
    result = convimg(image, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)

# Mean Filter
def mean(image, kernel_size=3):
    kernel_size = int(kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return convimg(image, kernel)

# Gaussian Filter
def gaussian(image, kernel_size=3, sigma=1.0):
    kernel_size = int(kernel_size)
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return convimg(image, kernel)

# Median Filter
def median(image, kernel_size=3):
    kernel_size = int(kernel_size)
    pad = kernel_size // 2
    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    H, W = image.shape
    output = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(region)

    return output

# Max Filter
def max(image, kernel_size=3):
    kernel_size = int(kernel_size)
    pad = kernel_size // 2
    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    H, W = image.shape
    output = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.max(region)

    return output