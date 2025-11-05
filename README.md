# Animal Body Condition Score (BCS) Prediction API

A Flask-based REST API for predicting animal body condition scores using deep learning models. The system combines YOLO object detection, view classification, and multi-view CNN models to assess animal body condition from images.

## Overview

This API provides three main functionalities:

1. **Animal Detection** - Detects and classifies animals in images using YOLO
2. **View Classification** - Identifies the viewing angle of animal images (Front, Back, Left, Right, Top)
3. **BCS Prediction** - Predicts Body Condition Score (1-9 scale) from multi-view images

### Body Condition Score Categories

- **TOO THIN**: BCS 1-3
- **IDEAL**: BCS 4-5
- **OBESE**: BCS 7-9

## Features

- Multi-view CNN architecture for accurate BCS prediction
- Automatic view angle detection using hierarchical classification
- RESTful API design with JSON responses

## Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install flask
pip install numpy
pip install pillow
pip install opencv-python
pip install ultralytics
pip install tensorflow
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### 4. Download Model Files (Google drive)

Create a models/ directory and download the pre-trained model files into it. The following Google Drive links host the model files — download each and save with the filenames shown.

#### Download Links

| Model File | Google Drive Link | Save As |
|------------|------------------|---------|
| Group View Classifier | [Download](https://drive.google.com/file/d/1Zt411ST93FLY47F9RjIiVU8FcAAaU6yM/view?usp=drive_link) | `models/FB_LR_T_EfficientNetB3.h5` |
| Left/Right Classifier | [Download](https://drive.google.com/file/d/1OhdsMDi9V8zFJV78rhGM47s_JmevciqO/view?usp=drive_link) | `models/L_R_EfficientNetB4.h5` |
| Front/Back Classifier | [Download](https://drive.google.com/file/d/1ME5LkUR3zlppdneFrzYcMECt-Jjsoa-f/view?usp=drive_link) | `models/F_B_DenseNet121.h5` |
| Multi-View BCS Model | [Download](https://drive.google.com/file/d/1sOOVUKQSzdHggq0in67U2VgYFLRbOW9n/view?usp=drive_link) | `models/bcs_model.h5` |


### 5. Run the Application

```bash
python app.py
```

The API will start on `http://127.0.0.1:5000` by default.

## API Endpoints

### 1. Animal Detection

**Endpoint:** `/predict-animal`  
**Method:** POST  
**Content-Type:** multipart/form-data

**Response:**
```json
{
  "prediction": "dog"
}
```

---

### 2. View Classification

**Endpoint:** `/predict-view`  
**Method:** POST  
**Content-Type:** multipart/form-data

**Response:**
```json
{
  "direction": "Left"
}
```

**Possible Directions:** Front, Back, Left, Right, Top

---

### 3. BCS Prediction

**Endpoint:** `/predict-bcs`  
**Method:** POST  
**Content-Type:** multipart/form-data

**Response:**
```json
{
  "bcs_score": 5,
  "bcs_category": "IDEAL"
}
```

## Model Architecture

### View Classification Pipeline

The view classification uses a hierarchical approach:

1. **Stage 1**: Group Classification (FrontBack / LeftRight / Top) - EfficientNetB3
2. **Stage 2a**: If LeftRight → Classify as Left or Right - EfficientNetB4
3. **Stage 2b**: If FrontBack → Classify as Front or Back - DenseNet121

### Multi-View BCS Model

The BCS prediction model uses a **Multi-View CNN** architecture that processes 4 different viewing angles:

1. **Top View** - Aerial view of the animal
2. **Back View** - Rear perspective
3. **Left View** - Left side profile
4. **Right View** - Right side profile

Each view is processed through separate CNN branches, and features are fused to make the final prediction.
