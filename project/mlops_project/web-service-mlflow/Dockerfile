# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Update package index and install necessary packages
RUN apt-get update \
    && apt-get install -y libgomp1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the test_model_ag directory into the container at /app/test_model_ag
COPY test_model_ag /app/test_model_ag

# Copy the predict.py file into the container at /app
COPY predict.py /app/

# Expose the port on which the app will run
EXPOSE 9696

# Command to run the application using Gunicorn
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
