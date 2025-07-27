FROM --platform=linux/amd64 python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY process_pdfs.py .
COPY heading_model_large8.txt* ./
COPY training_data_large8.csv* ./

# Create directories
RUN mkdir -p /app/input /app/output

# Run the processor
CMD ["python", "process_pdfs.py"]