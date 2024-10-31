import matplotlib.pyplot as plt
import detectree as dtr
import numpy as np
import cv2

def tree_detection_mask(img_path: str) -> np.ndarray:
    y_pred = dtr.Classifier().predict_img(img_path)
    return y_pred.astype(np.uint8)

def water_detection_mask(img_path: str, min_area_threshold: int = 500) -> np.ndarray:
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color Range
    lower_water = np.array([90, 50, 50])
    upper_water = np.array([140, 255, 255])

    mask_water = cv2.inRange(hsv, lower_water, upper_water)

    # Morphological operations (close small gaps in layer)
    water_kernel_threshold = 12
    water_kernel = np.ones((water_kernel_threshold, water_kernel_threshold), np.uint8)
    closed_water_mask = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, water_kernel)

    # Find contours in water segments
    contours, _ = cv2.findContours(closed_water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out artifacts (small water areas based on given threshold)
    filtered_water_mask = np.zeros_like(closed_water_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_threshold:
            cv2.drawContours(filtered_water_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return (filtered_water_mask > 0).astype(np.uint8)

def overlay_masks(original_image_path: str, tree_mask: np.ndarray, water_mask: np.ndarray) -> None:
    img = cv2.imread(original_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    combined_mask = np.zeros_like(img_rgb)
    combined_mask[tree_mask > 0] = (0, 255, 0) # Green RGB for trees
    combined_mask[water_mask > 0] = (0, 0, 255) # Blue RGB for water

    alpha_transparency = 0.5
    overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, combined_mask, alpha_transparency, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Tree + Water Mask Overlay")
    plt.show()

if __name__ == "__main__":
    image_input_path = "./main2.png"

    tree_mask = tree_detection_mask(image_input_path)
    water_mask = water_detection_mask(image_input_path)

    overlay_masks(image_input_path, tree_mask, water_mask)