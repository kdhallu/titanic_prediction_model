FROM python:3.9-slim

# Set the working directory
WORKDIR /model

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD flask --app predict run -h 0.0.0.0

#docker run -p 8090:5000 #imagename
