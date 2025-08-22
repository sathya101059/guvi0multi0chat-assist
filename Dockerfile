# Use official Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements.txt initially (for caching)
COPY requirements.txt .

# Install system packages needed to build some Python packages
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (including faiss-cpu)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Expose the port where Streamlit runs
EXPOSE 8501

# Command to launch your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
