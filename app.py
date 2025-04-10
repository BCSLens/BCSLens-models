import os
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load the model once
model = YOLO("yolov8n.pt")

# Upload folder (optional)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/yolo', methods=['POST'])
def yolo():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Read with OpenCV
    img = cv2.imread(image_path)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Run YOLO prediction
    results = model.predict(source=img)

    # Extract detection info
    output = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                output.append({
                    'class_id': cls_id,
                    'class_name': result.names[cls_id],
                    'confidence': round(confidence, 3)
                })

    return jsonify({'predictions': output})


if __name__ == '__main__':
    app.run(debug=True)
