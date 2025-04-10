# Use the official Node.js image
FROM node:18

# Set the working directory inside the container
WORKDIR /app

# Install Python and required dependencies
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create a Python virtual environment
RUN python3 -m venv /env

# Activate the virtual environment and install Python dependencies
RUN /env/bin/pip install --upgrade pip
COPY requirements.txt ./
RUN /env/bin/pip install -r requirements.txt

# Copy package.json and package-lock.json first to leverage Docker caching
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy the rest of the application files
COPY . .

# Expose the port that the app runs on
EXPOSE 5000

# Set the entry point to run both the Flask app and the YOLO inference script
CMD /env/bin/python yolo/yolo_inference.py & node index.js
