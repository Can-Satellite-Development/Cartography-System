# CivMapper: Automated Terrain Analysis and Infrastructure Planning
> Cartography System

This project provides a framework for analyzing terrain images and generating actionable masks for vegetation, water, and free areas. These masks are then used to place infrastructure elements such as buildings and paths. It incorporates advanced segmentation methods, including color-based masking, Gabor filters, and custom expansion techniques, to ensure accurate environmental detection.

---

## Related Content:
1. Tree Detection Library: https://github.com/ortanaV2/detectree?tab=readme-ov-file
4. Progress Updates (Discussion): https://github.com/orgs/Can-Satellite-Development/discussions/9

## Features

### 🌳 Tree Mask Detection
The **tree mask** identifies vegetation areas in an input image:
- Uses a tree classification model to predict vegetation regions.
- Expands the detected regions using contours and customizable thickness.
- Removes small artifacts based on a minimum area threshold.

### 🌊 Water Mask Detection
The **water mask** captures water bodies using:
- HSV color-based segmentation.
- Gabor filter integration to refine water detection based on texture.
- Iterative expansion of water regions while avoiding land boundaries.
- Small artifacts are filtered based on area thresholds.

### 🏞 Zero Mask (Free Area Detection)
The **zero mask** identifies areas free from vegetation and water:
- Combines tree and water masks.
- Generates an inverse mask to locate usable, open areas.

### 🌐 Coastal Mask and Inland Mask
- **Coastal Mask**: Detects areas near water bodies within a specific range.
- **Inland Mask**: Identifies areas away from water and coasts for interior infrastructure placement.

### 🌲 Forest Edge Mask
The **forest edge mask** captures areas near vegetation:
- Expands tree regions to identify areas suitable for edge-specific infrastructure.

### 🏗 Infrastructure Placement
The framework supports:
- Strategic placement of buildings based on terrain analysis.
- Generation of paths and bridges that navigate terrain obstacles.

---

## How It Works

### Masks and Their Purpose
1. **Tree Mask**: Detects vegetation zones.
2. **Water Mask**: Identifies water bodies and expands detection using texture analysis.
3. **Zero Mask**: Locates open areas free from water and trees.
4. **Coast Mask**: Highlights areas near water for ports and coastal structures.
5. **Inland Mask**: Finds interior regions for buildings and roads.
6. **Forest Edge Mask**: Identifies edges of forests for selective placement.

### Combining Masks
Masks are logically combined to define areas suitable for different types of infrastructure:
- Buildings are placed in inland or forest-edge areas.
- Roads avoid water and trees but connect key structures.
- Bridges span water bodies based on detected paths.

---

## Output Examples (Development Process)
![image](https://github.com/user-attachments/assets/fc2deaea-d6f2-4d62-8eb9-e939232a5348)
![image](https://github.com/user-attachments/assets/612fc5cb-6aef-4252-951a-60755f6ee357)
![image](https://github.com/user-attachments/assets/60609d98-5a04-4cb7-a7a4-b45d1609edb4)
![image](https://github.com/user-attachments/assets/ab2d3729-2893-48d0-96be-c83d74057697)




