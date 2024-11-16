import cv2
import tkinter as tk
import numpy as np
from tkinter import ttk
from area_mapping import overlay_mapping, get_tree_mask, get_water_mask
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def update_plot():
    overlay = img.copy()
    alpha: float = 0.5

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

    # Update matplot
    ax.clear()
    ax.set_facecolor((0.121, 0.121, 0.121))
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    canvas.draw()

img_path = "./mocking_examples/main4.png"
img = cv2.imread(img_path)

coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask = overlay_mapping(
    img_path, get_tree_mask(img_path), get_water_mask(img_path), dashboard=True
)

# Tkinter-Hauptfenster erstellen
root = tk.Tk()
root.title("Mask Dashboard")

sidebar = tk.Frame(root, width=250, bg="#c4c4c4")
sidebar.pack(side=tk.LEFT, fill=tk.Y)

coast_var = tk.BooleanVar(value=True)
inland_var = tk.BooleanVar(value=True)
forest_edge_var = tk.BooleanVar(value=True)
tree_var = tk.BooleanVar(value=True)
water_var = tk.BooleanVar(value=True)

ttk.Label(sidebar, text="Display Masks", font=("Arial", 12, "bold")).pack(pady=10)
ttk.Checkbutton(sidebar, text="Coast Mask", variable=coast_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Inland Mask", variable=inland_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Forest Edge Mask", variable=forest_edge_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Tree Mask", variable=tree_var, command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Water Mask", variable=water_var, command=update_plot).pack(anchor="w", padx=10, pady=5)

fig = Figure(figsize=(6, 4))
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

update_plot()

root.mainloop()