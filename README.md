# Animal BCS & View Prediction API

This project provides a Flask-based API for:
1. Identifying the animal in an image.
2. Predicting the view/camera angle of the animal (front, back, left, right, top).
3. Predicting the Body Condition Score (BCS) of an animal from four images (left, right, back, top).

It uses Ollama API with LLaVA and Bakllava models to analyze animal images.

## Features

* Upload a single image to identify the animal.
* Upload a single image to predict animal orientation (view).
* Upload 4 images (left, right, back, top) to predict BCS (1-9) and key contributing view.
* Responses include raw model output for debugging.

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/animal-bcs-api.git
cd animal-bcs-api
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
Flask==3.0.0
Pillow==10.1.0
requests==2.31.0
```

### 4. Install Ollama CLI

#### macOS

1. **ติดตั้ง Homebrew** (ถ้ายังไม่มี):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **ติดตั้ง Ollama CLI ผ่าน Homebrew**:
```bash
brew install ollama
```

3. **ตรวจสอบการติดตั้ง**:
```bash
ollama version
```

ถ้าเห็น version แสดงว่าติดตั้งสำเร็จแล้ว ✅

#### Windows / Linux

1. Go to [Ollama website](https://ollama.ai) and download the installer for your OS.
2. Install it according to your system instructions.
3. Verify installation:

```bash
ollama version
```

### 5. Install required models

After installing Ollama, download the models:

```bash
# Bakllava model (animal identification)
ollama pull bakllava:latest

# LLaVA model (view & BCS prediction)
ollama pull llava:latest
```

* Confirm the models are installed:

```bash
ollama list
```

### 6. Run Flask app

```bash
python app.py
```

* API will run at: `http://127.0.0.1:5000/`

## API Endpoints

### 1. `/predict` - Identify animal

* **Method:** `POST`
* **Form-data:** `image` (file)
* **Response:**

```json
{
  "prediction": "cat"
}
```

**Example:**
```bash
curl -X POST -F "image=@cat.jpg" http://127.0.0.1:5000/predict
```

---

### 2. `/predict_view` - Predict animal view

* **Method:** `POST`
* **Form-data:** `image` (file)
* **Response:**

```json
{
  "direction": "left",
  "raw_response": "<full model response>",
  "debug_cleaned": "<cleaned response>"
}
```

**Possible directions:**
- `front` - Animal is facing forward (face visible)
- `back` - Animal's rear/tail is visible
- `left` - Animal is facing/looking to the left
- `right` - Animal is facing/looking to the right
- `top` - Bird's eye view from above

**Example:**
```bash
curl -X POST -F "image=@cat.jpg" http://127.0.0.1:5000/predict_view
```

---

### 3. `/predict_bcs` - Predict Body Condition Score

* **Method:** `POST`
* **Form-data:** `left`, `right`, `back`, `top` (files)
* **Response:**

```json
{
  "bcs_score": 5,
  "bcs_category": "ideal",
  "reason": "Ribs can be felt but not prominently visible. Waist is clearly visible from top view.",
  "key_view": "top",
  "raw_response": "<full model response>"
}
```

**BCS Categories:**
- `1-3`: Underweight
- `4-5`: Ideal
- `6-7`: Overweight
- `8-9`: Obese

**Example:**
```bash
curl -X POST \
  -F "left=@left.jpg" \
  -F "right=@right.jpg" \
  -F "back=@back.jpg" \
  -F "top=@top.jpg" \
  http://127.0.0.1:5000/predict_bcs
```

## Project Structure

```
animal-bcs-api/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── static/
    └── uploads/          # Uploaded images (auto-created)
```

## Complete Code

**app.py:**
```python
from flask import Flask, request, jsonify
import requests
import base64
import os
from PIL import Image
import json
import re

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    """Identify the animal in the image"""
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

    # Encode image as base64
    with open(filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = "<image>\nWhat animal is in this image? Answer with only the name."

    # Call Ollama API with streaming
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
    """Predict which direction the animal is facing"""
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
        return jsonify({'error': 'Failed to process images', 'details': str(e)}), 500

    # Encode image as base64
    with open(filepath, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Prompt for direction prediction
    prompt = """Look at the image. There is an animal.

Question: Which way is the animal facing?

Choose ONLY the letter:
A = front
B = back
C = left
D = right
E = top

Answer (A/B/C/D/E):"""

    # Call Ollama API
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava:latest",
                "prompt": prompt,
                "images": [img_base64],
                "temperature": 0.1,
                "top_p": 0.9
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

    # Parse result
    view_prediction = result_text.strip().lower()
    
    # Map letter to direction
    letter_map = {
        'a': 'front',
        'b': 'back',
        'c': 'left',
        'd': 'right',
        'e': 'top'
    }
    
    # Get first character
    first_char = view_prediction[0] if view_prediction else None
    predicted_direction = letter_map.get(first_char, 'unknown')
    
    # Fallback: search for keywords
    if predicted_direction == 'unknown':
        direction_keywords = {
            'front': 'front',
            'back': 'back',
            'left': 'left',
            'right': 'right',
            'top': 'top'
        }
        
        for keyword, direction in direction_keywords.items():
            if keyword in view_prediction:
                predicted_direction = direction
                break

    return jsonify({
        "direction": predicted_direction,
        "raw_response": result_text.strip(),
        "debug_cleaned": view_prediction
    })


@app.route('/predict_bcs', methods=['POST'])
def predict_bcs():
    """Predict BCS from 4 views: left, right, back, top"""
    # Get all 4 images
    left_img = request.files.get('left')
    right_img = request.files.get('right')
    back_img = request.files.get('back')
    top_img = request.files.get('top')
    
    # Check if all images are provided
    if not all([left_img, right_img, back_img, top_img]):
        return jsonify({
            'error': 'Missing images',
            'details': 'Please provide all 4 views: left, right, back, top'
        }), 400

    # Process and save images
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
            # Save file
            filepath = os.path.join(UPLOAD_FOLDER, f"bcs_{view_name}_{img_file.filename}")
            img_file.save(filepath)
            
            # Resize image
            img = Image.open(filepath)
            img = img.convert("RGB")
            img = img.resize((512, 512))
            img.save(filepath)
            
            processed_images[view_name] = filepath
            
            # Encode as base64
            with open(filepath, "rb") as f:
                image_base64[view_name] = base64.b64encode(f.read()).decode("utf-8")
                
    except Exception as e:
        return jsonify({'error': 'Failed to process images', 'details': str(e)}), 500

    # Create BCS assessment prompt
    prompt = """You are a veterinary expert analyzing animal body condition.

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

    # Call Ollama API with all 4 images
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

    # Parse results
    bcs_score = None
    reason = ""
    key_view = ""
    
    # Find BCS score
    bcs_pattern = r'BCS[\s:]+(\d+(?:\.\d+)?)'
    bcs_match = re.search(bcs_pattern, result_text, re.IGNORECASE)
    if bcs_match:
        bcs_score = float(bcs_match.group(1))
    else:
        # Try to find any number between 1-9
        numbers = re.findall(r'\b([1-9])\b', result_text)
        if numbers:
            bcs_score = int(numbers[0])
    
    # Find Reason
    reason_pattern = r'Reason[\s:]+(.*?)(?=Key view|$)'
    reason_match = re.search(reason_pattern, result_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = result_text.strip()
    
    # Find Key view
    key_view_pattern = r'Key view[\s:]+(left|right|back|top)'
    key_view_match = re.search(key_view_pattern, result_text, re.IGNORECASE)
    if key_view_match:
        key_view = key_view_match.group(1).lower()
    
    # Determine BCS category
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
```

## Notes

* Ensure Ollama API is running locally at `http://127.0.0.1:11434/`.
* Images are resized to 512x512 before sending to the model.
* Use Postman or curl to test endpoints with image uploads.
* The BCS prediction works best with clear, well-lit images from all 4 angles.

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# macOS/Linux: restart the Ollama app
# Windows: restart Ollama from system tray
```

### Model not found
```bash
# Pull the models again
ollama pull bakllava:latest
ollama pull llava:latest
```

### Timeout errors
Increase the timeout in the code:
```python
timeout=180  # Change from 120 to 180 seconds
```

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.