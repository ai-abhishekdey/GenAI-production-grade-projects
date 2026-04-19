# Dockerfile

# Base Image
FROM python:3.11-slim

# Set Working Directory 
WORKDIR /app

# Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY src/   ./src/
COPY app.py .

# Expose Streamlit Port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
