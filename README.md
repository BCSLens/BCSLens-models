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