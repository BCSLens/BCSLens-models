from flask import Flask, request, jsonify, render_template
import requests
import base64
import os
from PIL import Image
import json

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

    # Encode image เป็น base64
    with open(filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = "<image>\nWhat animal is in this image? Answer with only the name."

    # ✅ เรียก Ollama API แบบ stream
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "bakllava:latest",
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
    prompt = prompt = """
    You are an expert animal image classifier.

    Task: Look at the uploaded image and identify the viewing direction of the animal based on which side of its body is visible.

    Definitions:

    front: animal facing the camera (head or face most visible)

    back: animal facing away from the camera (rear or tail most visible)

    left: animal’s head pointing left and tail to the right (left body side visible)

    right: animal’s head pointing right and tail to the left (right body side visible)

    top: animal viewed from above (back visible, legs on both sides)

    Respond with only one word: front, back, left, right, or top.
    """


    # เรียก Ollama API
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest",  # เปลี่ยนเป็น llava
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

    # ทำความสะอาดผลลัพธ์
    import re
    view_prediction = result_text.strip().lower()
    
    # Mapping keywords (เพิ่มคำที่หลากหลาย)
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

@app.route('/predict_bcs', methods=['POST'])
def predict_bcs():
    """
    Predict BCS (Body Condition Score) from any number of images.
    AI first predicts descriptive category (TOO THIN, IDEAL, OBESE),
    then maps it to numeric range: 1-3, 4-6, 7-9.
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

    # ✅ Prompt ให้ AI ตอบเป็นคำศัพท์
    prompt = "You are a veterinary expert. Look at the animal images and classify its body condition into ONE of these categories:\n\n"
    prompt += "TOO THIN: Ribs, spine, hips visible. Very little fat. Severe waist and abdominal tuck.\n"
    prompt += "IDEAL: Ribs can be felt but not visible. Waist visible. Slight abdominal tuck. Balanced body.\n"
    prompt += "OBESE: Ribs hard to feel. Heavy fat deposits. No waist or abdominal tuck.\n\n"
    prompt += "Answer with ONLY ONE of these words: TOO THIN, IDEAL, or OBESE.\n"
    prompt += "Your answer:"

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest",
                "prompt": prompt,
                "images": list(image_base64.values()),
                "temperature": 0.1,
                "num_predict": 15
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

    # ✅ Mapping AI keyword -> numeric range
    result_upper = result_text.strip().upper()
    mapping = {
        "TOO THIN": "1-3",
        "IDEAL": "4-6",
        "OBESE": "7-9"
    }
    bcs_range = mapping.get(result_upper, "4-6")  # default to IDEAL

    return jsonify({
        "bcs_category": result_upper,
        "bcs_range": bcs_range
    })

if __name__ == '__main__':
    app.run(debug=True)