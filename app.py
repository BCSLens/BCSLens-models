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
    prompt = """You are an expert in animal anatomy and image analysis.

    You will analyze the uploaded image and determine the view (camera angle) of the animal. Follow these steps carefully:

    Step 1: Identify the animal in the image.
    Step 2: Locate the head, tail, and main body orientation.
    Step 3: Determine which side of the animal is mostly visible.
    Step 4: Infer the camera angle relative to the animal based on head and tail orientation.

    Rules:
    - Only answer with ONE word: front, back, left, right, or top.
    - Do NOT include any explanation, extra words, or punctuation.
    - Focus on visible body features to make the decision.

    Answer:"""


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
    Predict BCS (Body Condition Score) from 4 views: left, right, back, top
    """
    # รับรูปภาพ 4 ด้าน
    left_img = request.files.get('left')
    right_img = request.files.get('right')
    back_img = request.files.get('back')
    top_img = request.files.get('top')
    
    # ตรวจสอบว่ามีรูปครบทั้ง 4 ด้าน
    if not all([left_img, right_img, back_img, top_img]):
        return jsonify({
            'error': 'Missing images', 
            'details': 'Please provide all 4 views: left, right, back, top'
        }), 400

    # บันทึกและ process รูปทั้ง 4 ด้าน
    processed_images = {}
    image_base64 = {}
    
    views = {
        'left': left_img,
        'right': right_img,
        'back': back_img,
        'top': top_img
    }
    
    try:
        for view_name, img_file in views.items():
            # บันทึกไฟล์
            filepath = os.path.join(UPLOAD_FOLDER, f"bcs_{view_name}_{img_file.filename}")
            img_file.save(filepath)
            
            # Resize รูปภาพ
            img = Image.open(filepath)
            img = img.convert("RGB")
            img = img.resize((512, 512))
            img.save(filepath)
            
            processed_images[view_name] = filepath
            
            # Encode เป็น base64
            with open(filepath, "rb") as f:
                image_base64[view_name] = base64.b64encode(f.read()).decode("utf-8")
                
    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'details': str(e)}), 500

    # สร้าง prompt สำหรับ BCS assessment
    prompt = f"""You are a veterinary expert analyzing animal body condition. 

I will show you 4 images of the same animal from different angles:
1. Left side view
2. Right side view  
3. Back view
4. Top view (from above)

Analyze the animal's body condition and determine the Body Condition Score (BCS) on a scale of 1-9:

BCS 1-3 (Underweight):
- Ribs, spine, and hip bones are very visible
- No fat coverage
- Severe muscle wasting

BCS 4-5 (Ideal):
- Ribs can be felt but not prominently visible
- Visible waist when viewed from above
- Slight abdominal tuck

BCS 6-7 (Overweight):
- Ribs difficult to feel
- Waist barely visible
- Abdominal fat present

BCS 8-9 (Obese):
- Ribs cannot be felt
- No waist visible
- Heavy fat deposits

Based on ALL 4 images, provide:
1. The BCS score (1-9)
2. Brief explanation of your assessment
3. Which view was most helpful

Format your answer as:
BCS: [number]
Reason: [explanation]
Key view: [left/right/back/top]"""

    # เรียก Ollama API พร้อมรูป 4 ด้าน
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest",
                "prompt": prompt,
                "images": [
                    image_base64['left'],
                    image_base64['right'],
                    image_base64['back'],
                    image_base64['top']
                ],
                "temperature": 0.3,
                "num_predict": 200
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

    # Parse ผลลัพธ์
    import re
    
    bcs_score = None
    reason = ""
    key_view = ""
    
    # หา BCS score
    bcs_pattern = r'BCS[\s:]+(\d+(?:\.\d+)?)'
    bcs_match = re.search(bcs_pattern, result_text, re.IGNORECASE)
    if bcs_match:
        bcs_score = float(bcs_match.group(1))
    else:
        # ลองหาตัวเลขที่อยู่ในช่วง 1-9
        numbers = re.findall(r'\b([1-9])\b', result_text)
        if numbers:
            bcs_score = int(numbers[0])
    
    # หา Reason
    reason_pattern = r'Reason[\s:]+(.*?)(?=Key view|$)'
    reason_match = re.search(reason_pattern, result_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # ถ้าไม่เจอ ใช้ข้อความทั้งหมด
        reason = result_text.strip()
    
    # หา Key view
    key_view_pattern = r'Key view[\s:]+(left|right|back|top)'
    key_view_match = re.search(key_view_pattern, result_text, re.IGNORECASE)
    if key_view_match:
        key_view = key_view_match.group(1).lower()
    
    # กำหนดระดับ BCS
    bcs_category = "unknown"
    if bcs_score:
        if bcs_score <= 3:
            bcs_category = "underweight"
        elif bcs_score <= 5:
            bcs_category = "ideal"
        elif bcs_score <= 7:
            bcs_category = "overweight"
        else:
            bcs_category = "obese"

    return jsonify({
        "bcs_score": bcs_score,
        "bcs_category": bcs_category,
        "reason": reason,
        "key_view": key_view,
        "raw_response": result_text.strip()
    })

if __name__ == '__main__':
    app.run(debug=True)