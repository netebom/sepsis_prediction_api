# Use an official Python 3.9 image as the base
FROM python:3.9-slim

# Set the working directory to here
WORKDIR .

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application code
COPY . .

# Set the command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]