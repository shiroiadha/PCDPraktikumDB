import numpy as np

def fourier_transform(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # Geser frekuensi nol ke tengah
    magnitude = np.abs(fshift)
    magnitude_spectrum = 20 * np.log(np.clip(magnitude, 1, None))  # Supaya bisa dilihat

    return magnitude_spectrum.astype(np.uint8)

def dft2d(image):
    image = image.astype(np.float32)
    M, N = image.shape
    dft_result = np.zeros((M, N), dtype=complex)

    for u in range(M):
        for v in range(N):
            sum_val = 0.0
            for x in range(M):
                for y in range(N):
                    angle = -2j * np.pi * ((u * x) / M + (v * y) / N)
                    sum_val += image[x, y] * np.exp(angle)
            dft_result[u, v] = sum_val

    magnitude = np.abs(dft_result)
    magnitude_spectrum = 20 * np.log(np.clip(magnitude, 1, None))

    return magnitude_spectrum.astype(np.uint8)