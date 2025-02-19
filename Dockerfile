# Use an official lightweight Python image.
FROM python:3.9-slim

# Set a working directory.
WORKDIR /

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port (matching the uvicorn port)
EXPOSE 8000

# Command to run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
