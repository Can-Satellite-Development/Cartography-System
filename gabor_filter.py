
import numpy as np
import cv2

def gabor_detection_mask(img_path: str, kernel_size: int = 31, sigma: float = 4.0, gamma: float = 0.5, psi: float = 0, orientations: list[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4], wavelengths: list[float] = [np.pi/4, np.pi/8, np.pi/12]) -> np.ndarray:
    img = cv2.imread(img_path)

    # Initialize Gabor filter parameters
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gabor_sum = np.zeros_like(gray_img, dtype=np.float32)

    # Apply multiple Gabor filters with varying parameters
    for theta in orientations:
        for lamda in wavelengths:
            gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
            filtered_img = cv2.filter2D(gray_img, cv2.CV_32F, gabor_kernel)
            gabor_sum += np.abs(filtered_img)  # Accumulate responses from each filter

    # Normalize the combined Gabor response and threshold to create a binary mask
    gabor_sum = cv2.normalize(gabor_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, gabor_mask = cv2.threshold(gabor_sum, 50, 255, cv2.THRESH_BINARY)  # Threshold value can be adjusted

    return gabor_mask
