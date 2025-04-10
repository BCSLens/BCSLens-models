FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up a virtual environment (optional but recommended)
RUN python3 -m venv /env

# Activate the virtual environment and install dependencies in it
RUN /env/bin/pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose port for Flask
EXPOSE 5000

# Start Flask app
CMD ["/env/bin/python3", "yolo/yolo_inference.py", "&", "/env/bin/flask", "run", "--host=0.0.0.0"]
