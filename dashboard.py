import cv2
import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from area_mapping import overlay_mapping, get_tree_mask, get_water_mask
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

# Function to update the plot based on the image
def update_plot(loading=False):
    overlay = img.copy()
    alpha: float = 0.35

    # Apply masks
    if coast_var.get():
        coast_color_overlay = np.zeros_like(img)
        coast_color_overlay[coast_mask > 0] = (174, 235, 52)[::-1]
        overlay = cv2.addWeighted(overlay, 1, coast_color_overlay, alpha, 0)

    if inland_var.get():
        inland_color_overlay = np.zeros_like(img)
        inland_color_overlay[inland_mask > 0] = (245, 130, 37)[::-1]
        overlay = cv2.addWeighted(overlay, 1, inland_color_overlay, alpha, 0)

    if forest_edge_var.get():
        forest_edge_color_overlay = np.zeros_like(img)
        forest_edge_color_overlay[forest_edge_mask > 0] = (81, 153, 14)[::-1]
        overlay = cv2.addWeighted(overlay, 1, forest_edge_color_overlay, alpha, 0)

    if tree_var.get():
        tree_color_overlay = np.zeros_like(img)
        tree_color_overlay[tree_mask > 0] = (66, 191, 50)[::-1]
        overlay = cv2.addWeighted(overlay, 1, tree_color_overlay, alpha, 0)

    if water_var.get():
        water_color_overlay = np.zeros_like(img)
        water_color_overlay[water_mask > 0] = (58, 77, 222)[::-1]
        overlay = cv2.addWeighted(overlay, 1, water_color_overlay, alpha, 0)

    # Update matplotlib plot
    ax.clear()

    # Show "Loading..." text when loading flag is set
    ax.text(0.5, 0.5, 'Loading...' if loading else "", color='black', fontsize=18, ha='center', va='center', transform=ax.transAxes)

    ax.set_facecolor((0.121, 0.121, 0.121))
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    # Display paths if path layer is enabled
    if path_var.get():
        for path_points in paths_points:
            if path_points is not None:
                for i, point in enumerate(path_points):
                    if i > 0:
                        x1, y1 = path_points[i - 1]
                        x2, y2 = point
                        line = plt.Line2D(
                            [x1, x2], [y1, y2],
                            linewidth=3,
                            color=(0.7, 0.7, 0.7) if point not in bridge_points else (0.8, 0.6, 0.4)
                        )
                        ax.add_line(line)  # Add the path line

    # Display buildings if building layer is enabled
    if building_var.get():
        for building in buildings:
            x, y, w, h = building["rect"]
            rect = plt.Rectangle(
                (x, y), w, h,
                linewidth=1, edgecolor="white", facecolor="none"
            )
            ax.add_patch(rect)  # Draw the building rectangle
            ax.text(
                x + w / 2, y - 5,
                building["nametag"],
                color="white", fontsize=6, ha="center"
            )

    canvas.draw()

# Function to load a new image based on the selection in the dropdown
def load_image(event=None):
    update_plot(loading=True)  # Show "Loading..." text before loading masks

    global img, coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points
    
    # Delay loading and calculating masks to ensure the "Loading..." text is shown
    def update_masks():
        global img, coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points
        
        # Load the selected image
        img_path = os.path.join("./mocking_examples", image_selection.get())
        img = cv2.imread(img_path)
        
        # Calculate masks for the new image
        coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points = overlay_mapping(
            img_path, get_tree_mask(img_path), get_water_mask(img_path), dashboard=True
        )
        
        # Update plot with new masks
        update_plot(loading=False)  # Remove "Loading..." text and show new masks

    # Delay mask loading using `after` to ensure the text is shown
    root.after(100, update_masks)

# Create Tkinter main window
root = tk.Tk()
root.title("Mask Dashboard")

# Sidebar
sidebar = tk.Frame(root, width=250, bg="#c4c4c4")
sidebar.pack(side=tk.LEFT, fill=tk.Y)

# Variables for masks
coast_var = tk.BooleanVar(value=True)
inland_var = tk.BooleanVar(value=True)
forest_edge_var = tk.BooleanVar(value=True)
tree_var = tk.BooleanVar(value=True)
water_var = tk.BooleanVar(value=True)
building_var = tk.BooleanVar(value=True)
path_var = tk.BooleanVar(value=True)

# Labels and checkbuttons for masks
ttk.Label(sidebar, text="Toggle Layers", font=("Arial", 12, "bold")).pack(pady=10)
ttk.Checkbutton(sidebar, text="Coast Mask", variable=coast_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Inland Mask", variable=inland_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Forest Edge Mask", variable=forest_edge_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Tree Mask", variable=tree_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Water Mask", variable=water_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Buildings", variable=building_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Paths", variable=path_var, command=update_plot).pack(anchor="w", padx=10, pady=5)

# Dropdown menu for selecting an image
image_files = [f for f in os.listdir("./mocking_examples") if f.endswith(".png")]
image_selection = ttk.Combobox(sidebar, values=image_files)
image_selection.pack(padx=10, pady=10)
image_selection.bind("<<ComboboxSelected>>", load_image)

# Load and display the default image
img_path = os.path.join("./mocking_examples", image_files[0])  # Select first image
img = cv2.imread(img_path)

# Calculate masks for the first image (predefine globals)
coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points = overlay_mapping(
    img_path, get_tree_mask(img_path), get_water_mask(img_path), dashboard=True
)

# Create matplotlib plot
fig = Figure(figsize=(6, 4))
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Initialize plot
update_plot()

root.mainloop()
