import numpy as np
import json
import cv2
import networkx as nx

def __plot_overlap__(rect1, rect2, min_distance):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (
        x1 + w1 + min_distance <= x2 or
        x2 + w2 + min_distance <= x1 or
        y1 + h1 + min_distance <= y2 or
        y2 + h2 + min_distance <= y1
    )

def get_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mask_range(mask: np.ndarray, contour_min_size: int = 1000, range_size: int = 200) -> np.ndarray:
    near_mask = np.zeros_like(mask) # new empty mask
    contours = get_contours(mask)

    # Add radius around contours to mask
    for cnt in contours:
        if cv2.contourArea(cnt) >= contour_min_size:
            cv2.drawContours(near_mask, [cnt], -1, 255, thickness=range_size)

    return near_mask

def get_mask_regions(mask: np.ndarray) -> list:
    # Apply connected components to label each separate area
    num_labels, labels = cv2.connectedComponents(mask)

    # Create a list to store individual masks
    mask_regions = []

    # Generate each mask for the labeled regions (ignore label 0, which is the background)
    for label in range(1, num_labels):
        # Create a new mask for each component
        component_mask = (labels == label).astype(np.uint8)
        mask_regions.append(component_mask)
    
    return mask_regions

def get_mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    # Calculate moments of the mask
    moments = cv2.moments(mask)

    # Calculate the centroid from the moments (cx, cy)
    if moments["m00"] != 0:  # To avoid division by zero
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # If the area is zero, set centroid to None
        return None
    
    return cx, cy

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

def place_buildings(blueprints: list, amounts: dict[str, int], masks: dict[str, np.ndarray]) -> list:
    # Get buildings from blueprints
    buildings_to_place = []
    for blueprint in blueprints:
        # Place amount of blueprint specified
        for _ in range(amounts[blueprint["name"]]):
            buildings_to_place.append(blueprint)

    # Place buildings
    placed_buildings = []
    for building in buildings_to_place:
        nametag = building["name"]
        dimensions = building["size"]
        location = building["location"]

        # Get possible location mask
        mask = masks[location]
        centroid = get_mask_centroid(mask)

        # Iterate over the mask
        for y, x in sorted(np.argwhere(mask > 0), key=lambda x: ((x[0] - centroid[0]) ** 2 + (x[1] - centroid[1]) ** 2) ** 0.5):  # Sort by distance to centroid
            rect_width, rect_height = dimensions[0], dimensions[1]

            #Check if rectangles fit within the mask-area
            if (x + rect_width <= mask.shape[1]) and (y + rect_height <= mask.shape[0]):
                if np.all(mask[y:y + rect_height, x:x + rect_width] > 0):
                    new_rect = (x, y, rect_width, rect_height)

                    # Check building collision
                    min_distance = 10
                    if all(not __plot_overlap__(new_rect, placed_building["rect"], min_distance) for placed_building in placed_buildings):
                        placed_buildings.append({"nametag": nametag, "rect": new_rect})
                        break
    
    return placed_buildings

def overlay_from_masks(img, *masks: np.ndarray) -> None:
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

# Custom Delaunay triangulation function
def custom_delaunay(points):
    edges = set()
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                pi, pj, pk = points[i], points[j], points[k]
                circumcenter, radius = get_circumcircle(pi, pj, pk)
                if circumcenter is None:
                    continue
                is_valid = True
                for m in range(num_points):
                    if m != i and m != j and m != k:
                        if np.linalg.norm(points[m] - circumcenter) < radius:
                            is_valid = False
                            break
                if is_valid:
                    edges.add((i, j))
                    edges.add((j, k))
                    edges.add((k, i))
    return edges

def get_circumcircle(p1, p2, p3):
    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    D = np.linalg.det(A)
    if D == 0:
        return None, np.inf
    A1 = np.array([
        [p1[0]**2 + p1[1]**2, p1[1], 1],
        [p2[0]**2 + p2[1]**2, p2[1], 1],
        [p3[0]**2 + p3[1]**2, p3[1], 1]
    ])
    A2 = np.array([
        [p1[0]**2 + p1[1]**2, p1[0], 1],
        [p2[0]**2 + p2[1]**2, p2[0], 1],
        [p3[0]**2 + p3[1]**2, p3[0], 1]
    ])
    A3 = np.array([
        [p1[0]**2 + p1[1]**2, p1[0], p1[1]],
        [p2[0]**2 + p2[1]**2, p2[0], p2[1]],
        [p3[0]**2 + p3[1]**2, p3[0], p3[1]]
    ])
    x = np.linalg.det(A1) / (2 * D)
    y = -np.linalg.det(A2) / (2 * D)
    radius = np.sqrt(np.linalg.det(A3) / D + x**2 + y**2)
    return np.array([x, y]), radius

def building_centers(buildings: list) -> np.ndarray:
    return np.array([(building["rect"][0] + building["rect"][2] // 2, building["rect"][1] + building["rect"][3] // 2) for building in buildings])

def generate_paths(buildings: list) -> list[tuple[tuple]]:
    centers = building_centers(buildings)

    # Generate paths between all centers using Delauney
    edges = custom_delaunay(centers)

    # Create a graph from the edges
    graph = nx.Graph()
    for edge in edges:
        p1, p2 = centers[edge[0]], centers[edge[1]]
        distance = np.linalg.norm(p1 - p2)
        graph.add_edge(edge[0], edge[1], weight=distance)

    # Create minimum spanning tree of graph
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    
    paths = []
    for edge in mst.edges():
        p1, p2 = centers[edge[0]], centers[edge[1]]
        paths.append((p1, p2))

    return paths