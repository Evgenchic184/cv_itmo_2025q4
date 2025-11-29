import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def dilate_opencv(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def dilate_native(image, kernel_size=(3, 3)):
    h, w = image.shape
    kh, kw = kernel_size
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.max(region)
    return output


def compare_performance(image, iterations=10):
    times_opencv = []
    times_native = []

    for _ in range(iterations):
        start = time.time()
        _ = dilate_opencv(image)
        times_opencv.append(time.time() - start)

        start = time.time()
        _ = dilate_native(image)
        times_native.append(time.time() - start)

    return times_opencv, times_native


if __name__ == "__main__":
    image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

    dilated_cv = dilate_opencv(image)
    dilated_native = dilate_native(image)

    print(f"diff by methods: {abs(dilated_cv - dilated_native).sum()}")

    cv2.imwrite('dilated_opencv.png', dilated_cv)
    cv2.imwrite('dilated_native.png', dilated_native)

    times_cv, times_native = compare_performance(image)
    print(f"OpenCV dilation: {np.mean(times_cv):.6f} ± {np.std(times_cv):.6f} sec")
    print(f"Native dilation: {np.mean(times_native):.6f} ± {np.std(times_native):.6f} sec")



