from flask import Flask, request, jsonify, render_template
import requests
import base64
import os
from PIL import Image
import json
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Save original file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Resize image
    try:
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = img.resize((512, 512))
        img.save(filepath)
    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500

    # Encode image to base64
    with open(filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = "<image>\nWhat animal is in this image? Answer with only the word: dog or cat or cow or sheep or horse."

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest",
                "prompt": prompt,
                "images": [img_base64]
            },
            stream=True,
            timeout=120
        )

        result_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if "response" in data:
                        result_text += data["response"]
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return jsonify({'error': 'Failed to call Ollama API', 'details': str(e)}), 500

    return jsonify({"prediction": result_text.strip()})


@app.route('/predict_view', methods=['POST'])
def predict_view():
    """
    Predict which direction the animal is facing (top, front, back, right, left)
    """
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Save original file
    filepath = os.path.join(UPLOAD_FOLDER, f"view_{file.filename}")
    file.save(filepath)

    # Resize image
    try:
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = img.resize((512, 512))
        img.save(filepath)
    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500

    # Encode image เป็น base64
    with open(filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Prompt แบบ step-by-step
    prompt = (
        "<image>\n"
        "Please identify the **viewing direction** of the animal in this image.\n"
        "Definitions:\n"
        "- front: the animal is facing toward the camera; the head or face is most visible.\n"
        "- back: the animal is facing away from the camera; the tail or rear is most visible.\n"
        "- left: the animal’s head points to the left and tail to the right (left side visible).\n"
        "- right: the animal’s head points to the right and tail to the left (right side visible).\n"
        "- top: the animal is viewed from above; the back is visible and all four legs may appear at both sides.\n\n"
        "Answer with **only one word** — exactly one of these: front, back, left, right, or top.\n"
        "Do not include explanations, punctuation, or confidence levels."
    )

    # Call Ollama API
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest", 
                "prompt": prompt,
                "images": [img_base64],
                "temperature": 0.3,
                "num_predict": 50  # จำกัดจำนวนคำตอบ
            },
            stream=True,
            timeout=120
        )

        result_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if "response" in data:
                        result_text += data["response"]
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return jsonify({'error': 'Failed to call Ollama API', 'details': str(e)}), 500

    # Clean the response and map to valid directions
    import re
    view_prediction = result_text.strip().lower()
    
    direction_mapping = {
        'front': 'front',
        'frontal': 'front',
        'face': 'front',
        'facing': 'front',
        'forward': 'front',
        'towards': 'front',
        'back': 'back',
        'rear': 'back',
        'behind': 'back',
        'tail': 'back',
        'backside': 'back',
        'backward': 'back',
        'left': 'left',
        'leftward': 'left',
        'left side': 'left',
        'to the left': 'left',
        'pointing left': 'left',
        'right': 'right',
        'rightward': 'right',
        'right side': 'right',
        'to the right': 'right',
        'pointing right': 'right',
        'top': 'top',
        'above': 'top',
        'overhead': 'top',
        'bird': 'top',
        'aerial': 'top'
    }
    
    valid_directions = ['front', 'back', 'left', 'right', 'top']
    predicted_direction = None
    
    # วิธีที่ 1: หาคำสุดท้ายในประโยค (มักเป็นคำตอบ)
    words = view_prediction.split()
    for word in reversed(words):
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in valid_directions:
            predicted_direction = clean_word
            break
    
    # วิธีที่ 2: ค้นหาจาก mapping
    if not predicted_direction:
        for keyword, direction in direction_mapping.items():
            if keyword in view_prediction:
                predicted_direction = direction
                break
    
    # วิธีที่ 3: หาคำที่อยู่หลัง "answer:" หรือ "answer is"
    if not predicted_direction:
        answer_pattern = r'(?:answer|direction|facing)[\s:is]+(\w+)'
        match = re.search(answer_pattern, view_prediction)
        if match:
            answer_word = match.group(1)
            if answer_word in valid_directions:
                predicted_direction = answer_word
    
    # Default
    if not predicted_direction:
        predicted_direction = "unknown"

    return jsonify({
        "direction": predicted_direction,
        "raw_response": result_text.strip(),
        "debug_cleaned": view_prediction
    })

model2_group_view_path = 'models/FB_LR_T_EfficientNetB3.h5'
model2_left_right_view_path = 'models/L_R_EfficientNetB4.h5'
model2_front_back_view_path = 'models/F_B_DenseNet121.h5'

model2_group_view = load_model(model2_group_view_path)
model2_left_right_view = load_model(model2_left_right_view_path)
model2_front_back_view = load_model(model2_front_back_view_path)

group_view_class_names = ['FrontBack', 'LeftRight', 'Top']
left_right_class_names = ['Left', 'Right']
front_back_class_names = ['Back', 'Front']

def prepare_image_from_bytes(img_bytes, target_size):
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
        processed_img = prepare_image_from_bytes(img_bytes, target_size=(300, 300))
        prediction = model2_group_view.predict(processed_img)
        predicted_class = group_view_class_names[np.argmax(prediction)]
        if predicted_class == 'LeftRight':
            processed_img_2 = prepare_image_from_bytes(img_bytes, target_size=(380, 380))
            prediction = model2_left_right_view.predict(processed_img_2)
            predicted_class = left_right_class_names[np.argmax(prediction)]
        elif predicted_class == 'FrontBack':
            processed_img_3 = prepare_image_from_bytes(img_bytes, target_size=(224, 224))
            prediction = model2_front_back_view.predict(processed_img_3)
            predicted_class = front_back_class_names[np.argmax(prediction)]
        return jsonify({'direction': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_bcs', methods=['POST'])
def predict_bcs():
    """
    Predict BCS (Body Condition Score) from any number of images.
    AI predicts specific BCS score 1-9 based on detailed veterinary criteria.
    """
    uploaded_files = request.files
    if not uploaded_files:
        return jsonify({
            'error': 'No images uploaded',
            'details': 'Please provide at least one image'
        }), 400
    
    processed_images = {}
    image_base64 = {}
    
    try:
        for view_name, img_file in uploaded_files.items():
            filepath = os.path.join(UPLOAD_FOLDER, f"bcs_{view_name}_{img_file.filename}")
            img_file.save(filepath)
            img = Image.open(filepath).convert("RGB")
            img = img.resize((512, 512))
            img.save(filepath)
            processed_images[view_name] = filepath
            
            with open(filepath, "rb") as f:
                image_base64[view_name] = base64.b64encode(f.read()).decode("utf-8")
                
    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'details': str(e)}), 500
    
    prompt = """You are a veterinary expert specializing in Body Condition Scoring (BCS). 
    Analyze the animal images and assign a BCS score from 1 to 9 based on these detailed criteria:

    **BCS 1 (EMACIATED):**
    - Ribs, spine, and pelvic bones are extremely visible and prominent
    - No body fat detectable
    - Severe muscle wasting
    - Waist and abdominal tuck are extremely pronounced

    **BCS 2 (VERY THIN):**
    - Ribs, spine, and pelvic bones are easily visible
    - Minimal body fat
    - Obvious muscle wasting
    - Severe waist and abdominal tuck

    **BCS 3 (THIN):**
    - Ribs are easily felt and visible with no pressure
    - Top of spine is visible
    - Pelvic bones becoming prominent
    - Obvious waist and abdominal tuck

    **BCS 4 (UNDERWEIGHT):**
    - Ribs are easily felt with minimal fat covering
    - Waist is clearly visible from above
    - Abdominal tuck is evident
    - Minimal fat on tail base

    **BCS 5 (IDEAL):**
    - Ribs can be felt without excess pressure, but not visible
    - Waist is visible when viewed from above
    - Slight abdominal tuck when viewed from side
    - Well-proportioned body

    **BCS 6 (OVERWEIGHT):**
    - Ribs are felt with slight pressure due to fat covering
    - Waist is barely visible from above
    - Minimal abdominal tuck
    - Fat deposits noticeable at tail base

    **BCS 7 (HEAVY):**
    - Ribs are difficult to feel under fat covering
    - No waist visible, back appears broadened
    - Little to no abdominal tuck
    - Obvious fat deposits on tail base and hips

    **BCS 8 (OBESE):**
    - Ribs cannot be felt under heavy fat covering
    - No waist, abdomen is distended
    - No abdominal tuck, may have sagging abdomen
    - Prominent fat deposits on tail base, hips, and chest

    **BCS 9 (SEVERELY OBESE):**
    - Massive fat deposits on chest, spine, tail base, and hips
    - Ribs completely obscured by heavy fat
    - Waist completely absent
    - Abdomen is significantly distended
    - Difficulty in movement may be visible

    Carefully examine all provided images and respond with ONLY ONE NUMBER from 1 to 9 that best represents the animal's body condition.

    Your answer (single digit only):"""

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest",
                "prompt": prompt,
                "images": list(image_base64.values()),
                "temperature": 0.1,
                "num_predict": 5,  # เพิ่มความแม่นยำในการตอบเป็นตัวเลข
                "top_p": 0.9,
                "top_k": 10
            },
            stream=True,
            timeout=180
        )
        
        result_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if "response" in data:
                        result_text += data["response"]
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        return jsonify({'error': 'Failed to call Ollama API', 'details': str(e)}), 500
    
    # ✅ Extract numeric BCS from response
    result_cleaned = result_text.strip()
    
    # Extract first digit found in response
    import re
    bcs_match = re.search(r'[1-9]', result_cleaned)
    
    if bcs_match:
        bcs_score = int(bcs_match.group())
        # Validate BCS is in range 1-9
        if bcs_score < 1 or bcs_score > 9:
            bcs_score = 5  # Default to ideal if invalid
    else:
        bcs_score = 5  # Default to ideal if no number found
    
    return jsonify({
        "bcs_score": bcs_score
    })

CLASS_NAMES = ["TOO THIN", "IDEAL", "OBESE"]
RANGE_MAPPING = {
        "TOO THIN": "1-3",
        "IDEAL": "4-5",
        "OBESE": "6-9"
    }

MODEL_PATH = "models/best_bcs_model.h5"
model_bcs = load_model(MODEL_PATH)
@app.route('/predict_bcs2', methods=['POST'])
def predict_bcs2():
    """
    Predict exact BCS (1–9) using trained Multi-View CNN model (.h5)
    Accepts multiple images (views): top, back, left, right
    Uses placeholder black image if any view is missing.
    """
    uploaded_files = request.files
    if not uploaded_files:
        return jsonify({'error': 'No images uploaded'}), 400

    # Placeholder image for missing views
    placeholder = np.zeros((224, 224, 3), dtype=np.float32)

    # Expected view names
    views = ['top', 'back', 'left', 'right']
    processed_views = {v: placeholder for v in views}

    try:
        for view_name, img_file in uploaded_files.items():
            v = view_name.lower()
            if v not in views:
                continue  # skip unknown view names

            filepath = os.path.join(UPLOAD_FOLDER, f"bcs_{v}_{img_file.filename}")
            img_file.save(filepath)

            img = Image.open(filepath).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            processed_views[v] = img_array.astype(np.float32)

    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'details': str(e)}), 500

    # Stack into batch dimension (batch_size=1)
    input_batch = [np.expand_dims(processed_views[v], 0) for v in views]

    try:
        preds = model_bcs.predict(input_batch)
        avg_pred = preds[0]  # shape: (9,) for 9 classes (BCS 1–9)
        class_idx = np.argmax(avg_pred)  # 0-based index
        bcs_score = class_idx + 1  # convert to 1–9 scale
        confidence = float(np.max(avg_pred))
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    # Optionally, categorize (if you still want labels)
    if 1 <= bcs_score <= 3:
        bcs_category = "TOO THIN"
    elif 4 <= bcs_score <= 6:
        bcs_category = "IDEAL"
    else:
        bcs_category = "OBESE"

    return jsonify({
        "bcs_score": int(bcs_score),
        "bcs_category": bcs_category,
        "confidence": confidence,
        "raw_output": avg_pred.tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)