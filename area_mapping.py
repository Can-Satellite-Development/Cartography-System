import matplotlib.pyplot as plt
import detectree as dtr
import numpy as np
import cv2
import json

def get_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def tree_detection_mask(img_path: str, expansion_thickness: int = 2, min_area: int = 10) -> np.ndarray:
    y_pred = dtr.Classifier().predict_img(img_path)
    tree_mask = y_pred.astype(np.uint8)

    contours = get_contours(tree_mask)

    # Draw Contours around vegetation areas based on "expansion-thickness"
    expanded_mask = np.zeros_like(tree_mask) # new mask layer
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.fillPoly(expanded_mask, [cnt], 255)

            if expansion_thickness > 0:
                cv2.drawContours(expanded_mask, [cnt], -1, 255, thickness=expansion_thickness)

    return expanded_mask

def water_detection_mask(img_path: str, min_area_threshold: int = 500, water_kernel_size: int = 12) -> np.ndarray:
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color Range
    lower_water = np.array([90, 50, 50])
    upper_water = np.array([140, 255, 255])

    mask_water = cv2.inRange(hsv, lower_water, upper_water)

    # Morphological operations (close small gaps in layer)
    water_kernel = np.ones((water_kernel_size, water_kernel_size), np.uint8)
    closed_water_mask = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, water_kernel)

    # Find contours in water segments
    contours = get_contours(closed_water_mask)

    # Filter out artifacts (small water areas based on given threshold)
    filtered_water_mask = np.zeros_like(closed_water_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_threshold:
            cv2.drawContours(filtered_water_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return (filtered_water_mask > 0).astype(np.uint8)

def near_water_mask(zero_mask: np.ndarray, water_mask: np.ndarray, water_source_min_size: int = 1000, coast_range: int = 200) -> np.ndarray:
    coast_mask = np.zeros_like(zero_mask) # new empty mask
    contours = get_contours(water_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= water_source_min_size:
            cv2.drawContours(coast_mask, [cnt], -1, 255, thickness=coast_range)
    
    coast_mask = np.logical_and(zero_mask > 0, coast_mask > 0).astype(np.uint8)
    return coast_mask

def rectangles_overlap(rect1, rect2, min_distance):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (
        x1 + w1 + min_distance <= x2 or
        x2 + w2 + min_distance <= x1 or
        y1 + h1 + min_distance <= y2 or
        y2 + h2 + min_distance <= y1
    )

def get_buildings(sort_priority: bool = True) -> list:
    with open("buildings.json", "r") as f:
        buildings = json.load(f)
    
    if sort_priority:
        # Sorting by priority (descending) and then by size (descending)
        return sorted(
            buildings, 
            key=lambda x: (x["priority"], x["size"][0] * x["size"][1]), 
            reverse=True
            )
    else:
        return buildings

def place_buildings(mask: np.ndarray, building_blueprints: list, amounts: dict[str, int]) -> list:
    # Get buildings from blueprints
    buildings_to_place = []
    for blueprint in building_blueprints:
        # Place amount of blueprint specified
        for _ in range(amounts[blueprint["name"]]):
            buildings_to_place.append(blueprint)

    # Place buildings
    placed_buildings = []
    for building in buildings_to_place:
        nametag = building["name"]
        dimensions = building["size"]

        # Iterate over the mask
        for x, y in np.argwhere(mask > 0): # x, y for positive mask points
            rect_width, rect_height = dimensions[0], dimensions[1]

            #Check if rectangles fit within the mask-area
            if (x + rect_width <= mask.shape[1]) and (y + rect_height <= mask.shape[0]):
                if np.all(mask[y:y + rect_height, x:x + rect_width] > 0):
                    new_rect = (x, y, rect_width, rect_height)

                    # Check building collision
                    min_distance = 10
                    if all(not rectangles_overlap(new_rect, placed_building["rect"], min_distance) for placed_building in placed_buildings):
                        placed_buildings.append({"nametag": nametag, "rect": new_rect})
                        break

    return placed_buildings

def overlay_mapping(img_path: str, tree_mask: np.ndarray, water_mask: np.ndarray, min_area_threshold: int = 2500) -> None:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ## Masks

    # Combine tree and water masks to find free areas
    nature_mask = np.logical_or(tree_mask > 0, water_mask > 0).astype(np.uint8)
    
    free_area_mask = (nature_mask == 0).astype(np.uint8)  # Inverted mask to get free areas

    contours = get_contours(free_area_mask)

    # Create a new mask to keep only large free areas
    zero_mask = np.zeros_like(free_area_mask)

    # Iterate over contours and filter out small free areas based on min-area-threshold (-> in pixels)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_threshold:
            # Draw the contour on the cleaned mask if it is large enough
            cv2.drawContours(zero_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    coast_mask = near_water_mask(zero_mask, water_mask)

    # Display the result
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))

    ## Buildings

    blueprints = get_buildings()
    buildings = place_buildings(zero_mask, blueprints, {"res-building 1": 1, "res-building 2": 1, "workshop": 1, "HEP-Plant": 2})
    
    ## Overlays
    
    alpha_transparency = 0.5

    # Water Overlay
    water_overlay = img_rgb.copy()
    water_overlay[water_mask > 0] = (0, 0, 255)  # Blue RGB for water
    water_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, water_overlay, alpha_transparency, 0)

    # Tree Overlay
    tree_overlay = img_rgb.copy()
    tree_overlay[tree_mask > 0] = (0, 255, 0)   # Green RGB for trees
    tree_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, tree_overlay, alpha_transparency, 0)

    # Develop Vegetation/Water Overlay
    nature_overlay = img_rgb.copy()
    nature_overlay[water_mask > 0] = (0, 0, 255)  # Blue RGB for water
    nature_overlay[tree_mask > 0] = (0, 255, 0)   # Green RGB for trees
    nature_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, nature_overlay, alpha_transparency, 0)

    # Coast Overlay
    coast_overlay = img_rgb.copy()
    coast_overlay[coast_mask > 0] = (255, 255, 0)  # Yellow RGB for coast
    coast_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, coast_overlay, alpha_transparency, 0)

    # Overlay the free areas on the original image
    zero_overlay = img_rgb.copy()
    zero_overlay[zero_mask == 255] = (255, 0, 0)  # Red RGB for free areas
    zero_overlay = cv2.addWeighted(img_rgb, 1 - alpha_transparency, zero_overlay, alpha_transparency, 0)

    # Building Overlay
    building_overlay = img_rgb.copy()

    ## Display

    axes[0][0].imshow(water_overlay)
    axes[0][0].set_title("Water Overlay")

    axes[0][1].imshow(tree_overlay)
    axes[0][1].set_title("Tree Overlay")

    axes[0][2].imshow(nature_overlay)
    axes[0][2].set_title("Nature Overlay")

    axes[1][0].imshow(coast_overlay)
    axes[1][0].set_title("Coast Overlay (Coast Range: 200px)")

    axes[1][1].imshow(zero_overlay)
    axes[1][1].set_title("Zero Overlay")

    axes[1][2].imshow(building_overlay)
    axes[1][2].set_title("Building Overlay")
    for building in buildings:
        x, y, w, h = building["rect"]
        nametag = building["nametag"]
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        axes[1][2].add_patch(rect)

        axes[1][2].text(x + w/2, y - 5, nametag, color="red", fontsize=6, ha="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_input_path = "./mocking-examples/main2.png"

    tree_mask = tree_detection_mask(image_input_path)
    water_mask = water_detection_mask(image_input_path)

    overlay_mapping(image_input_path, tree_mask, water_mask)