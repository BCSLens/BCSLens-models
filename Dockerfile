# Use the official Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip python3-venv

# Create a Python virtual environment
RUN python3 -m venv /env

# Activate the virtual environment and upgrade pip
RUN /env/bin/pip install --upgrade pip

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt ./
RUN /env/bin/pip install -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the entry point to run the Flask app and the YOLO inference script
CMD /env/bin/python yolo/yolo_inference.py & flask run --host=0.0.0.0
