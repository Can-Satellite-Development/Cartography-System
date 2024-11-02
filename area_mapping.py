
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

def mask_range(mask: np.ndarray, contour_min_size: int = 1000, range: int = 200) -> np.ndarray:
    near_mask = np.zeros_like(mask) # new empty mask
    contours = get_contours(mask)

    # Add radius around contours to mask
    for cnt in contours:
        if cv2.contourArea(cnt) >= contour_min_size:
            cv2.drawContours(near_mask, [cnt], -1, 255, thickness=range)
    
    return near_mask

def combine_masks(*masks: np.ndarray, operation: np.ufunc = np.logical_and) -> np.ndarray:
    combined_mask = masks[0]

    for mask in masks[1:]:
        combined_mask = operation(combined_mask, mask > 0).astype(np.uint8)
    
    return combined_mask

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

def overlay_from_masks(img_path: str, *masks: tuple[np.ndarray, list[int, int, int], float]) -> None:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Add each mask with its color to the Overlay
    overlay = img_rgb.copy()
    for mask, color, alpha in masks:
        mask_overlay = img_rgb.copy()
        mask_overlay[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1 - alpha, mask_overlay, alpha, 0)
        # Blend the overlay with the original image
    overlay = cv2.addWeighted(img_rgb, 0, overlay, 1, 0)

    return overlay

def filter_artifacts(mask: np.ndarray, min_area_threshold: int = 2500) -> np.ndarray:
    contours = get_contours(mask)
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_threshold:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask

def overlay_mapping(img_path: str, tree_mask: np.ndarray, water_mask: np.ndarray, min_area_threshold: int = 2500) -> None:

    ## Masks
    
    water_range_mask = mask_range(water_mask, contour_min_size=1000, range=100)

    tree_range_mask = mask_range(tree_mask, contour_min_size=500, range=50)

    nature_mask = combine_masks(tree_mask, water_mask, operation=np.logical_or)
    
    free_area_mask = (nature_mask == 0).astype(np.uint8)  # Inverted mask to get free areas
    zero_mask = filter_artifacts(free_area_mask, min_area_threshold=min_area_threshold)

    coast_mask = combine_masks(water_range_mask, zero_mask, operation=np.logical_and)

    forest_edge_mask = combine_masks(tree_range_mask, zero_mask, operation=np.logical_and)

    ## Buildings

    blueprints = get_buildings()
    buildings = place_buildings(zero_mask, blueprints, {"res-building 1": 1, "res-building 2": 1, "workshop": 1, "HEP-Plant": 2})
    
    ## Overlays
    
    alpha_transparency = 0.5
    range_alpha_transparency = 0.2

    # Water Overlay
    water_overlay = overlay_from_masks(img_path, (water_mask, (0, 0, 255), alpha_transparency), (water_range_mask, (0, 0, 255), range_alpha_transparency))

    # Tree Overlay
    tree_overlay = overlay_from_masks(img_path, (tree_mask, (0, 255, 0), alpha_transparency), (tree_range_mask, (0, 255, 0), range_alpha_transparency))

    # Nature Overlay
    nature_overlay = overlay_from_masks(img_path, (water_mask, (0, 0, 255), alpha_transparency), (tree_mask, (0, 255, 0), alpha_transparency))

    # Coast Overlay
    coast_overlay = overlay_from_masks(img_path, (coast_mask, (255, 0, 255), alpha_transparency))

    # Forest Edge Overlay
    forest_edge_overlay = overlay_from_masks(img_path, (forest_edge_mask, (255, 255, 0), alpha_transparency))

    # Free Area Overlay
    zero_overlay = overlay_from_masks(img_path, (zero_mask, (255, 0, 0), alpha_transparency))

    ## Display

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))

    axes[0][0].imshow(water_overlay)
    axes[0][0].set_title("Water & Water Range Overlay")

    axes[0][1].imshow(tree_overlay)
    axes[0][1].set_title("Tree & Tree Range Overlay")

    axes[0][2].imshow(nature_overlay)
    axes[0][2].set_title("Nature Overlay")

    axes[1][0].imshow(coast_overlay)
    axes[1][0].set_title("Coast Overlay")

    axes[1][1].imshow(forest_edge_overlay)
    axes[1][1].set_title("Forest Edge Overlay")

    axes[1][2].imshow(zero_overlay)
    axes[1][2].set_title("Zero & Building Overlay")
    building_color = (0.1, 0, 0)
    for building in buildings:
        x, y, w, h = building["rect"]
        nametag = building["nametag"]
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor=building_color, facecolor="none")
        axes[1][2].add_patch(rect)

        axes[1][2].text(x + w/2, y - 5, nametag, color=building_color, fontsize=6, ha="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_input_path = "./mocking-examples/main2.png"

    tree_mask = tree_detection_mask(image_input_path)
    water_mask = water_detection_mask(image_input_path)

    overlay_mapping(image_input_path, tree_mask, water_mask)
