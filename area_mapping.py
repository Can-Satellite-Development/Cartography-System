import matplotlib.pyplot as plt
import utilities as utils
import detectree as dtr
import numpy as np
import cv2

def get_tree_mask(img_path: str, expansion_thickness: int = 2, min_area: int = 10) -> np.ndarray:
    y_pred = dtr.Classifier().predict_img(img_path)
    tree_mask = y_pred.astype(np.uint8)

    contours = utils.get_contours(tree_mask)

    # Draw Contours around vegetation areas based on "expansion-thickness"
    expanded_mask = np.zeros_like(tree_mask)  # new mask layer
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.fillPoly(expanded_mask, [cnt], 1)  # Debug: Changed to 1 instead of 255

            if expansion_thickness > 0:
                cv2.drawContours(expanded_mask, [cnt], -1, 1, thickness=expansion_thickness)  # Debug: Changed to 1 instead of 255

    return expanded_mask

def get_water_mask(img_path: str, min_area_threshold: int = 500, water_kernel_size: int = 12, radius: float = 3) -> np.ndarray:
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color Range
    lower_water = np.array([90, 50, 50])
    upper_water = np.array([140, 255, 255])

    mask_water = cv2.inRange(hsv, lower_water, upper_water)

    # Morphological operations (close small gaps in layer)
    water_kernel = np.ones((water_kernel_size, water_kernel_size), np.uint8)
    closed_water_mask = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, water_kernel)

    filtered_water_mask = utils.filter_artifacts(closed_water_mask, min_area_threshold=min_area_threshold)

    scale_factor: float = 0.35

    water_mask =  utils.scale_mask((filtered_water_mask > 0).astype(np.uint8), scale_factor)
    gabor_filter_mask = utils.scale_mask(get_gabor_filter_mask(img), scale_factor)

    # Expands color detected water mask based on gabor filter mask 
    iterations_amount = 8
    for i in range(iterations_amount):
        y_coords, x_coords = np.where(water_mask == 1)
        for y, x in zip(y_coords, x_coords):
            # Get gabor filter mask values from given radius
            radius_values: list = utils.get_values_in_radius(mask=gabor_filter_mask, coords=(x, y), radius=radius)
            # Check if collision with land occurs
            if not sum(radius_values) >= 1:
                # Expand radius if smooth area detected
                water_mask = utils.set_radius(mask=water_mask, coords=(x, y), radius=radius, value=1)

    water_mask = utils.filter_artifacts(water_mask, min_area_threshold=min_area_threshold)

    return cv2.resize(water_mask, tuple(reversed(filtered_water_mask.shape)), interpolation=cv2.INTER_NEAREST)

def get_zero_mask(tree_mask: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    # Combine tree and water masks to find free areas
    combined_mask = np.logical_or(tree_mask > 0, water_mask > 0).astype(np.uint8)

    # Inverted mask to get free areas
    zero_mask = (combined_mask == 0).astype(np.uint8)

    return zero_mask

def get_gabor_filter_mask(img, ksize=15, sigma=4.3, theta=0.2, lambd=9.8, gamma=0.2, psi=0.885) -> np.ndarray:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gabor kernel (filter)
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

    # Apply gabor filter to the image
    gabor_mask = cv2.filter2D(gray_img, cv2.CV_8UC3, gabor_kernel)

    return gabor_mask

def get_coast_mask(zero_mask: np.ndarray, water_mask: np.ndarray, water_source_min_size: int = 1000, coast_range: int = 100) -> np.ndarray:
    coast_mask = utils.mask_range(water_mask, contour_min_size=water_source_min_size, range_size=coast_range)
    
    coast_mask = np.logical_and(zero_mask > 0, coast_mask > 0).astype(np.uint8)
    return coast_mask

def get_inland_mask(zero_mask: np.ndarray, coast_mask: np.ndarray) -> np.ndarray:
    # Convert masks to binary masks
    zero_mask = (zero_mask > 0).astype(np.uint8) * 1  # Debug: Changed to 1 instead of 255
    coast_mask = (coast_mask > 0).astype(np.uint8) * 1  # Debug: Changed to 1 instead of 255

    return cv2.bitwise_and(zero_mask, cv2.bitwise_not(coast_mask))

def get_forest_edge_mask(tree_mask: np.ndarray, zero_mask: np.ndarray, contour_min_size: int = 500, range_size: int = 50) -> np.ndarray:
    tree_range_mask = utils.mask_range(tree_mask, contour_min_size=contour_min_size, range_size=range_size)
    forest_edge_mask = np.logical_and(tree_range_mask, zero_mask).astype(np.uint8)
    
    return forest_edge_mask

def mask_percentages(water_mask: np.ndarray, zero_mask: np.ndarray, tree_mask: np.ndarray) -> tuple[float, float, float]:
    total_pixels = water_mask.size

    water_percentage = round((np.sum(water_mask == 1) / total_pixels) * 100, 2)
    zero_percentage = round((np.sum(zero_mask == 1) / total_pixels) * 100, 2)
    tree_percentage = round((np.sum(tree_mask == 1) / total_pixels) * 100, 2)

    # Priority weights
    water_weight = 15
    building_weight = 40
    tree_weight = 45

    tolerance_bonus = 0

    for percentage in [(water_percentage, water_weight), (zero_percentage, building_weight), (tree_percentage, tree_weight)]:
        if abs(percentage[0] - percentage[1]) <= 10:
            tolerance_bonus += 10
        else:
            if percentage != (water_percentage, water_weight):
                tolerance_bonus -= 5

    if zero_percentage < 10:
        tolerance_bonus -= 10
        
    if water_percentage < 5:
        tolerance_bonus -= 20

    score = (water_percentage * water_weight + zero_percentage * building_weight + tree_percentage * tree_weight + tolerance_bonus) / 100
    score = min(100, round(score * 1.75 + tolerance_bonus, 2))

    return (water_percentage, zero_percentage, tree_percentage, score)

def mask_deployment(tree_mask: np.ndarray, water_mask: np.ndarray, costs: tuple[int] = (1000, 3500, 5000, 10000), image_height: float = 250) -> tuple[np.ndarray]:
    zero_mask = get_zero_mask(tree_mask, water_mask)
    utils.paste_debugging("zero mask generated")  #* Debugging (Time Paste)

    # Against fully by one mask enclosed zones, specifically artifacts from tree detection
    utils.switch_enclaves(zero_mask, tree_mask, water_mask, enclosed_by_one=True, enclave_size_threshold=2500)
    utils.paste_debugging("remove enclave artifacts threshold=2500 (True)")  #* Debugging (Time Paste)

    # Against all artifacts, much smaller threshold as to only get rid of small artifacts and not actually useful areas
    utils.switch_enclaves(zero_mask, tree_mask, water_mask, enclosed_by_one=False, enclave_size_threshold=500)
    utils.paste_debugging("remove enclave artifacts threshold=500 (False)")  #* Debugging (Time Paste)

    coast_mask = get_coast_mask(zero_mask, water_mask)
    utils.paste_debugging("coast mask generated")  #* Debugging (Time Paste)
    inland_mask = get_inland_mask(zero_mask, coast_mask)
    utils.paste_debugging("inland mask generated")  #* Debugging (Time Paste)
    forest_edge_mask = get_forest_edge_mask(tree_mask, zero_mask)
    utils.paste_debugging("forest edge mask generated")  #* Debugging (Time Paste)
    water_and_coast_mask = np.logical_or(water_mask == 1, coast_mask == 1).astype(np.uint8)
    utils.paste_debugging("water & coast mask generated")  #* Debugging (Time Paste)

    blueprints = utils.get_buildings()
    utils.paste_debugging("blueprint received")  #* Debugging (Time Paste)

    scaling_factor = 250 / image_height    

    buildings, building_mask = utils.place_buildings(blueprints, 
                                    masks={
                                        "zero": zero_mask,
                                        "coast": coast_mask, 
                                        "inland": inland_mask, 
                                        "forest_edge": forest_edge_mask, 
                                        "water_and_coast": water_and_coast_mask},
                                    scaling_factor = scaling_factor
                                    )
    utils.paste_debugging("buildings placed")  #* Debugging (Time Paste)
    
    # Generate List of paths
    paths_points, bridge_points = utils.generate_path_points(buildings, masks_and_cost_multipliers={
        "zero": (zero_mask, costs[0]), 
        "trees": (tree_mask, costs[1]), 
        "water": (water_mask, costs[2]), 
        "buildings": (building_mask, costs[3]),  # Buildings must be avoided at all costs
    }, resolution_factor=0.35, max_distance=None)  # Generate paths using masks scaled down to 35%, with a maximum distance between points of e.g. 100 pixels (deactivated by None)
    utils.paste_debugging("paths points generated")  #* Debugging (Time Paste)

    percentages = mask_percentages(water_mask, zero_mask, tree_mask)
    utils.paste_debugging("habitability calculated")  #* Debugging (Time Paste)

    return (coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points, percentages)

#* Mocking Dashboard for Performance Measuring
if __name__ == "__main__":
    image_input_path = "./mocking_examples/main2.png"

    utils.paste_debugging("start with dataset load")  #* Debugging (Time Paste)
    tree_mask = get_tree_mask(image_input_path)
    utils.paste_debugging("tree mask generated")  #* Debugging (Time Paste)
    water_mask = get_water_mask(image_input_path)
    utils.paste_debugging("water mask generated")  #* Debugging (Time Paste)

    result_tuple = mask_deployment(tree_mask, water_mask)
