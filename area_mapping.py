
import matplotlib.pyplot as plt
import numpy as np
import cv2
from gabor_filter import gabor_detection_mask
from tree_filter import tree_detection_mask
from water_filter import water_detection_mask


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
