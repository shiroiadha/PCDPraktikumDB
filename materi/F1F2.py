import numpy as np
import cv2

def convimg(image, kernel):
    Hk, Wk = kernel.shape
    pad_h = Hk // 2
    pad_w = Wk // 2

    # Tambahan padding jika ukuran kernel genap
    pad_bottom = pad_h if Hk % 2 == 1 else pad_h + 1
    pad_right = pad_w if Wk % 2 == 1 else pad_w + 1

    padded_img = cv2.copyMakeBorder(image, pad_h, pad_bottom, pad_w, pad_right, cv2.BORDER_REFLECT)
    HImg, WImg = image.shape[:2]
    convout = np.zeros((HImg, WImg), dtype=np.float32)

    for i in range(HImg):
        for j in range(WImg):
            region = padded_img[i:i+Hk, j:j+Wk]
            convout[i, j] = np.sum(region * kernel)

    return np.clip(convout, 0, 255).astype(np.uint8)

def sobel(image):
    Sx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    Gx = convimg(image, Sx)
    Gy = convimg(image, Sy)

    magnitude = np.sqrt(Gx.astype(float) ** 2 + Gy.astype(float) ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def prewitt(image):
    Px = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
    Py = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

    Gx = convimg(image, Px)
    Gy = convimg(image, Py)

    magnitude = np.sqrt(Gx.astype(float) ** 2 + Gy.astype(float) ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def roberts(image):
    Rx = np.array([[1, 0],
                   [0, -1]])
    Ry = np.array([[0, 1],
                   [-1, 0]])

    Gx = convimg(image, Rx)
    Gy = convimg(image, Ry)

    magnitude = np.sqrt(Gx.astype(float) ** 2 + Gy.astype(float) ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def laplacian(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    result = convimg(image, kernel)
    return result


def log(image):
    kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])
    result = convimg(image, kernel)
    return result



# CANNY SECTION ONLY
def canny(image, kernel_size=5, sigma=1.4, low_threshold=50, high_threshold=150):
    # 1. Grayscale (optional kalau gambar udah grayscale)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # 3. Sobel Gradients
    Gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(Gx, Gy)
    direction = np.arctan2(Gy, Gx) * 180 / np.pi
    direction = (direction + 180) % 180

    # 4. Non-Maximum Suppression
    nms = non_maximum_suppression(magnitude, direction)

    # 5. Thresholding & Hysteresis
    edges = hysteresis_thresholding(nms, low_threshold, high_threshold)

    return edges.astype(np.uint8)


def non_maximum_suppression(magnitude, direction):
    H, W = magnitude.shape
    output = np.zeros((H, W), dtype=np.float32)

    angle = direction.copy()
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255

            angle_deg = angle[i, j]

            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= angle_deg < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= angle_deg < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= angle_deg < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output


def hysteresis_thresholding(img, low_thresh, high_thresh):
    H, W = img.shape
    res = np.zeros((H, W), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img <= high_thresh) & (img >= low_thresh))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Hysteresis
    for i in range(1, H-1):
        for j in range(1, W-1):
            if res[i, j] == weak:
                if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                    or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                    or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res