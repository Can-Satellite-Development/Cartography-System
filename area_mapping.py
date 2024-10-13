import cv2
import numpy as np

img = cv2.imread('./mocking examples/mocking_example1.png') # image input
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color boundaries/range for vegetation
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 200]) 

lower_orange = np.array([10, 100, 100])
upper_orange = np.array([20, 255, 255])

lower_brown = np.array([5, 50, 50])
upper_brown = np.array([20, 200, 150])

# Color boundaries for water sources
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
def filter_small_contours(contours, min_area):
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

min_area_threshold = 200
contours_vegetation = filter_small_contours(contours_vegetation, min_area_threshold)
contours_water = filter_small_contours(contours_water, min_area_threshold)

result_img = img.copy() # Copy for visual

# Fill colors for vegetation and water
green_fill_color = (0, 255, 0)
blue_fill_color = (255, 0, 0)
alpha = 0.3 # Opacity

def fill_area_with_alpha(image, contours, color, alpha):
    overlay = image.copy()
    cv2.fillPoly(overlay, contours, color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# Fill out vegetation and water areas
fill_area_with_alpha(result_img, contours_vegetation, green_fill_color, alpha)
fill_area_with_alpha(result_img, contours_water, blue_fill_color, alpha)

def draw_contours(image, contours, color):
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.polylines(image, [approx], True, color, 1)

# Draw vegetation and water contours
draw_contours(result_img, contours_vegetation, green_fill_color)
draw_contours(result_img, contours_water, blue_fill_color) # Overlap vegetation

cv2.imshow("Cartography", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
