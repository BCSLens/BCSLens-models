import os
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Load the model once
model1 = YOLO("yolo12n.pt")

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
    results = model1.predict(source=img)

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

model2_group_view_path = 'final/mobilenet_LR_FB_T_88.h5'
model2_left_right_view_path = 'final/mobilenet_L_R_70.h5'
model2_front_back_view_path = 'final/mobilenet_F_B_70.h5'

model2_group_view = load_model(model2_group_view_path)
model2_left_right_view = load_model(model2_left_right_view_path)
model2_front_back_view = load_model(model2_front_back_view_path)

group_view_class_names = ['LeftRight', 'FrontBack', 'Top']
left_right_class_names = ['Left', 'Right']
front_back_class_names = ['Front', 'Back']

def prepare_image_from_bytes(img_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/classify-view', methods=['POST'])
def classify_view_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = img_file.read()
        processed_img = prepare_image_from_bytes(img_bytes)
        prediction = model2_group_view.predict(processed_img)
        predicted_class = group_view_class_names[np.argmax(prediction)]
        if predicted_class == 'LeftRight':
            prediction = model2_left_right_view.predict(processed_img)
            predicted_class = left_right_class_names[np.argmax(prediction)]
        elif predicted_class == 'FrontBack':
            prediction = model2_front_back_view.predict(processed_img)
            predicted_class = front_back_class_names[np.argmax(prediction)]
        return jsonify({'group': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

model3_bcs_path = 'final/bcs_prediction_model_97.h5'
model3_bcs = load_model(model3_bcs_path)
bcs_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

@app.route('/predict-bcs', methods=['POST'])
def predict_bcs_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = img_file.read()
        processed_img = prepare_image_from_bytes(img_bytes)
        prediction = model3_bcs.predict(processed_img)
        predicted_class = bcs_classes[np.argmax(prediction)]
        return jsonify({'bcs': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
