# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set the environment variable to indicate the app's environment
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
