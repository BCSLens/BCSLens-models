import cv2
from glob import glob
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# Load the model once
model = YOLO("yolo12n.pt")

# Dataset path
root_dir = 'dataset_evaluate'
classes = ['Dog', 'Cat', 'Horse', 'Sheep', 'Cow']
views = ['front', 'left', 'right', 'top', 'back']
image_paths = []

# Load image paths and labels
for cls in classes:
    for view in views:
        paths = glob(f"{root_dir}/{cls.lower()}/{view}/*.jpg")
        image_paths += [(img_path, cls) for img_path in paths] 
    print(f"Loaded {len([p for p in image_paths if p[1] == cls])} images for class '{cls}'")
    cls_dir = os.path.join(root_dir, cls.lower())
    print(f"Directory for {cls.lower()} {'exists' if os.path.exists(cls_dir) else 'does NOT exist.'}")

print(f"Total images to evaluate: {len(image_paths)}")

# Initialize lists for predictions and ground truth
y_true = []
y_pred = []
inference_times = []

# Start evaluation
print("Starting model evaluation...")
total_start_time = time.time()

for i, (img_path, true_label) in enumerate(image_paths):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        continue
    
    # Measure inference time for this image
    img_start_time = time.time()
    results = model.predict(img, conf=0.25, verbose=False)
    img_end_time = time.time()
    
    img_inference_time = img_end_time - img_start_time
    inference_times.append(img_inference_time)
    
    # Get prediction
    boxes = results[0].boxes
    if len(boxes) > 0:
        cls_id = int(boxes.cls[0].item())
        pred_label = model.names[cls_id].capitalize()
    else:
        pred_label = 'Unknown'  # Changed from 'none' to 'Unknown'
    
    y_true.append(true_label)
    y_pred.append(pred_label)
    
    # Print progress every 50 images
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(image_paths)} images")

total_end_time = time.time()

# Calculate metrics
total_inference_time = total_end_time - total_start_time
avg_inference_time_per_image = np.mean(inference_times)
total_model_inference_time = sum(inference_times)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Print results
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Total images evaluated: {len(y_true)}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Total evaluation time: {total_inference_time:.2f} seconds")
print(f"Total model inference time: {total_model_inference_time:.4f} seconds")
print(f"Average inference time per image: {avg_inference_time_per_image*1000:.2f} ms")
print(f"FPS (Frames Per Second): {1/avg_inference_time_per_image:.2f}")

# Show prediction distribution
print(f"\nPrediction distribution:")
unique_preds, counts = np.unique(y_pred, return_counts=True)
for pred, count in zip(unique_preds, counts):
    print(f"  {pred}: {count} predictions")

# Create confusion matrix
print("\nGenerating confusion matrix...")
all_labels = list(set(y_true + y_pred))  # Include all labels that appear
cm = confusion_matrix(y_true, y_pred, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("YOLOv12 Confusion Matrix")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.tight_layout()
plt.show()

# Additional metrics per class
print("\nPer-class accuracy:")
for cls in classes:
    cls_indices = [i for i, label in enumerate(y_true) if label == cls]
    if cls_indices:
        cls_predictions = [y_pred[i] for i in cls_indices]
        cls_accuracy = sum([1 for pred in cls_predictions if pred == cls]) / len(cls_predictions)
        print(f"  {cls}: {cls_accuracy * 100:.2f}% ({sum([1 for pred in cls_predictions if pred == cls])}/{len(cls_predictions)})")

print("\n" + "="*50)