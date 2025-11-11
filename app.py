from flask import Flask, request, jsonify
import io
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)

model_yolo = YOLO("yolo12n.pt")

model2_group_view_path = 'models/FB_LR_T_EfficientNetB3.h5'
model2_left_right_view_path = 'models/L_R_EfficientNetB4.h5'
model2_front_back_view_path = 'models/F_B_MobileNetV2.h5'

model2_group_view = load_model(model2_group_view_path)
model2_left_right_view = load_model(model2_left_right_view_path)
model2_front_back_view = load_model(model2_front_back_view_path)

group_view_class_names = ['FrontBack', 'LeftRight', 'Top']
left_right_class_names = ['Left', 'Right']
front_back_class_names = ['Back', 'Front']

model_bcs = load_model("models/bcs_model_MobileNetV2.h5")

def prepare_image_from_bytes(img_bytes, target_size):
    """Convert raw image bytes to preprocessed NumPy array."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/predict-animal', methods=['POST'])
def predict_animal():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()

    # Decode image from bytes
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Run YOLO prediction
    results = model_yolo.predict(source=img)

    best_class = None
    best_conf = 0.0

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_class = result.names[cls_id]

    if best_class is None:
        return jsonify({'prediction': 'No object detected'}), 200

    return jsonify({'prediction': best_class})

@app.route('/predict-view', methods=['POST'])
def predict_view():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_file = request.files['image']
    img_bytes = img_file.read()

    try:
        # 1st model: group classification (FrontBack / LeftRight / Top)
        processed_img = prepare_image_from_bytes(img_bytes, target_size=(300, 300))
        prediction = model2_group_view.predict(processed_img)
        predicted_class = group_view_class_names[np.argmax(prediction)]

        # Sub-models for finer classification
        if predicted_class == 'LeftRight':
            processed_img = prepare_image_from_bytes(img_bytes, target_size=(380, 380))
            prediction = model2_left_right_view.predict(processed_img)
            predicted_class = left_right_class_names[np.argmax(prediction)]

        elif predicted_class == 'FrontBack':
            processed_img = prepare_image_from_bytes(img_bytes, target_size=(224, 224))
            prediction = model2_front_back_view.predict(processed_img)
            predicted_class = front_back_class_names[np.argmax(prediction)]

        return jsonify({'direction': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-bcs', methods=['POST'])
def predict_bcs():
    uploaded_files = request.files
    if not uploaded_files:
        return jsonify({'error': 'No images uploaded'}), 400

    # Placeholder for missing views
    placeholder = np.zeros((224, 224, 3), dtype=np.float32)
    views = ['top', 'back', 'left', 'right']
    processed_views = {v: placeholder for v in views}

    try:
        for view_name, img_file in uploaded_files.items():
            v = view_name.lower()
            if v not in views:
                continue

            img_bytes = img_file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            processed_views[v] = img_array.astype(np.float32)

    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'details': str(e)}), 500

    # Combine to batch
    input_batch = [np.expand_dims(processed_views[v], 0) for v in views]

    try:
        preds = model_bcs.predict(input_batch)
        avg_pred = preds[0]
        class_idx = np.argmax(avg_pred)
        bcs_score = class_idx + 1
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    if 1 <= bcs_score <= 3:
        bcs_category = "TOO THIN"
    elif 4 <= bcs_score <= 5:
        bcs_category = "IDEAL"
    else:
        bcs_category = "OBESE"

    return jsonify({
        "bcs_score": int(bcs_score),
        "bcs_category": bcs_category
    })

if __name__ == '__main__':
    app.run(debug=True)
