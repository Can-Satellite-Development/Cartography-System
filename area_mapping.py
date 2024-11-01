
import matplotlib.pyplot as plt
import numpy as np
import cv2
import detectree as dtr



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



def overlay_mapping(img_path: str, tree_mask: np.ndarray, water_mask: np.ndarray, gabor_mask: np.ndarray, min_area_threshold: int = 2500) -> None:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    alpha_transparency = 0.5


    # Overlay Trees
    tree_overlay = img_rgb.copy()
    tree_overlay[tree_mask > 0] = (0, 255, 0)  # Green RGB for trees
    tree_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, tree_overlay, alpha_transparency, 0)


    # Overlay Water
    water_overlay = img_rgb.copy()
    water_overlay[water_mask > 0] = (0, 0, 255)  # Blue RGB for water
    water_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, water_overlay, alpha_transparency, 0)


    # Overlay Gabor
    gabor_overlay = img_rgb.copy()
    gabor_overlay[gabor_mask > 0] = (255, 0, 0)  # Red RGB for gabor
    gabor_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, gabor_overlay, alpha_transparency, 0)


    # Overlay Vegetation/Water Mask
    nature_overlay = img_rgb.copy()
    nature_overlay[water_mask > 0] = (0, 0, 255)  # Blue RGB for water
    nature_overlay[tree_mask > 0] = (0, 255, 0)   # Green RGB for trees
    nature_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, nature_overlay, alpha_transparency, 0)


    # Overlay Vegetation/Water/Gabor Mask
    combined_overlay = img_rgb.copy()
    combined_overlay[water_mask > 0] = (0, 0, 255)  # Blue RGB for water
    combined_overlay[tree_mask > 0] = (0, 255, 0)   # Green RGB for trees
    combined_overlay[gabor_mask > 0] = (255, 0, 0)   # Red RGB for gabor
    combined_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, combined_overlay, alpha_transparency, 0)


    # Combine tree and water masks to find free areas
    nature_mask = np.logical_or(tree_mask > 0, water_mask > 0).astype(np.uint8)
    
    free_area_mask = (nature_mask == 0).astype(np.uint8)  # Inverted mask to get free areas

    contours, _ = cv2.findContours(free_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask to keep only large free areas
    cleaned_free_area_mask = np.zeros_like(free_area_mask)

    # Iterate over contours and filter out small free areas based on min-area-threshold (-> in pixels)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_threshold:
            # Draw the contour on the cleaned mask if it is large enough
            cv2.drawContours(cleaned_free_area_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Overlay the free areas on the original image
    building_overlay = img_rgb.copy()
    building_overlay[cleaned_free_area_mask == 255] = (255, 255, 0)  # Yellow RGB for free areas
    building_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, building_overlay, alpha_transparency, 0)

    # Display the result
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))

    axes[0][0].imshow(tree_overlay)
    axes[0][0].set_title('Tree Overlay')

    axes[0][1].imshow(water_overlay)
    axes[0][1].set_title('Water Overlay')

    axes[0][2].imshow(gabor_overlay)
    axes[0][2].set_title('Gabor Overlay')

    axes[1][0].imshow(nature_overlay)
    axes[1][0].set_title('Tree + Water Mask Overlay')

    axes[1][1].imshow(combined_overlay)
    axes[1][1].set_title('Tree + Water + Gabor Mask Overlay')

    axes[1][2].imshow(building_overlay)
    axes[1][2].set_title('Free Building Areas Mask Overlay')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    image_input_path = "./mocking-examples/main2.png"

    tree_mask = tree_detection_mask(image_input_path)
    water_mask = water_detection_mask(image_input_path)
    gabor_mask = gabor_detection_mask(image_input_path, orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4], wavelengths=[np.pi/4, np.pi/8, np.pi/12])

    overlay_mapping(image_input_path, tree_mask, water_mask, gabor_mask)
