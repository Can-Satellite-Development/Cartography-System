import matplotlib.pyplot as plt
import detectree as dtr
import numpy as np
import cv2

def tree_detection_mask(img_path: str, expansion_thickness: int = 2, min_area: int = 10) -> np.ndarray:
    y_pred = dtr.Classifier().predict_img(img_path)
    tree_mask = y_pred.astype(np.uint8)

    contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours around vegetation areas based on "expansion-thickness"
    expanded_mask = np.zeros_like(tree_mask) # new mask layer
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.fillPoly(expanded_mask, [cnt], 255)

            if expansion_thickness > 0:
                cv2.drawContours(expanded_mask, [cnt], -1, 255, thickness=expansion_thickness)

    return expanded_mask

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

# Filter out small contours based on area (clear artifacts)
min_area_threshold = 200
contours_vegetation = [cnt for cnt in contours_vegetation if cv2.contourArea(cnt) > min_area_threshold]
contours_water = [cnt for cnt in contours_water if cv2.contourArea(cnt) > min_area_threshold]

# Create tree density heatmap
y_pred = dtr.Classifier().predict_img(input_image)
cleaned_heatmap = binary_opening(y_pred, structure=np.ones((3, 3))) # Remove artifacts

overlay = img.copy()
green_fill_color = tuple(reversed((55, 130, 40))) # RGB
blue_fill_color = tuple(reversed((36, 73, 138))) # RGB
alpha = 0.5  # Opacity for vegetation/water areas

def fill_area_with_alpha(image, contours, color, alpha):
    overlay = image.copy()
    cv2.fillPoly(overlay, contours, color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

# Fill out vegetation and water areas
result_img = fill_area_with_alpha(img, contours_vegetation, green_fill_color, alpha)
result_img = fill_area_with_alpha(result_img, contours_water, blue_fill_color, alpha)

# Overlay the tree density heatmap onto the result image
tree_density_color = (0, 100, 0) # RGB

# Convert cleaned heatmap into an 8-bit image for proper blending
tree_density_heatmap = np.uint8(cleaned_heatmap * 255)  # Rescale to 0-255
tree_density_colored = cv2.applyColorMap(tree_density_heatmap, cv2.COLORMAP_HOT)

# Combine the tree density heatmap with the vegetation/water map
final_overlay = cv2.addWeighted(result_img, 1, tree_density_colored, 0.25, 0)

cv2.imshow("Final Cartography with Tree Density Heatmap", final_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
