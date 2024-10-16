import cv2
import numpy as np
import detectree as dtr
from scipy.ndimage import binary_opening
import os

input_image = './mocking examples/test_input_6.png' # Input Image Path
img = cv2.imread(input_image)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color boundaries/range for vegetation
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 200])
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([20, 255, 255])
lower_brown = np.array([5, 50, 50])
upper_brown = np.array([20, 200, 150])
lower_water = np.array([80, 50, 20])
upper_water = np.array([130, 255, 200])

mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
mask_water = cv2.inRange(hsv, lower_water, upper_water)

# Combine vegetation masks (green, orange, brown)
combined_vegetation_mask = cv2.bitwise_or(mask_green, mask_orange)
combined_vegetation_mask = cv2.bitwise_or(combined_vegetation_mask, mask_brown)

# Morphological operations to merge/close small gaps in areas (veg./water) -> get big chunks
veg_kernel_threshold = 26
vegetation_kernel = np.ones((veg_kernel_threshold, veg_kernel_threshold), np.uint8)
closed_vegetation_mask = cv2.morphologyEx(combined_vegetation_mask, cv2.MORPH_CLOSE, vegetation_kernel)

water_kernel_threshold = 35
water_kernel = np.ones((water_kernel_threshold, water_kernel_threshold), np.uint8)
closed_water_mask = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, water_kernel)

# Find contours of vegetation and water areas
contours_vegetation, _ = cv2.findContours(closed_vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_water, _ = cv2.findContours(closed_water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

def save_image_with_unique_name(image, base_filename, extension='png'):
    # Start with base number 0
    counter = 0
    
    # Create unique filename
    filename = f"output examples/{base_filename}_{counter}.{extension}"
    
    # Count up until we find an unused filename
    while os.path.exists(filename):
        counter += 1
        filename = f"output examples/{base_filename}_{counter}.{extension}"
    
    # Save the image
    cv2.imwrite(filename, image)
    print(f"Saved final image to {filename}.")

# Save the final image
save_image_with_unique_name(final_overlay, 'test_output')

# Display the final image
cv2.imshow("Final Cartography with Tree Density Heatmap", final_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
