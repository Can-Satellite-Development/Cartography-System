
import detectree as dtr
import cv2
import numpy as np

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
