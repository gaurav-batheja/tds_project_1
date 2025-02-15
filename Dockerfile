# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /task-agent-api

# Copy and install dependencies separately for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the FastAPI server port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
