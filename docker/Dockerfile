# Base Image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account.json"

# Run the application
CMD ["python", "src/main.py", "--config", "config/config.yaml"]
