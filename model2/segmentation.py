import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the model
model = YOLO("yolo11n-seg.pt")  # or model = YOLO("path/to/best.pt")

# Predict on an image
results = model("cat.jpg")  # or your local image path

# Load the image for drawing
image = cv2.imread("cat.jpg")  # or use your local image path

# Create a blank image for the second column (only lines)
image_blank = np.zeros_like(image)

# Access and draw masks on both images
for result in results:
    # Extract polygon coordinates
    xy = result.masks.xy

    # Draw the polygon lines on the image
    for polygon in xy:
        # Convert the polygon from list of points to numpy array
        points = np.array(polygon, dtype=np.int32)

        # Draw the polygon on the original image (first column)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the polygon lines on the blank image (second column)
        cv2.polylines(image_blank, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# Convert the images from BGR to RGB (for displaying with matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_blank_rgb = cv2.cvtColor(image_blank, cv2.COLOR_BGR2RGB)

# Display the images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Image with mask and lines
axes[0].imshow(image_rgb)
axes[0].axis('off')  # Hide axes
axes[0].set_title("Image with Lines")

# Image with only lines (blank background)
axes[1].imshow(image_blank_rgb)
axes[1].axis('off')  # Hide axes
axes[1].set_title("Only Lines (No Image)")

plt.show()