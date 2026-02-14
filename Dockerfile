FROM python:3.8-slim-buster

# Prevent python from buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y awscli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port (optional but good practice)
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
