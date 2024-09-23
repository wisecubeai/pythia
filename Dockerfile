#FROM python:3.10-slim-bullseye
#
#WORKDIR /app
#COPY . /app
#
#RUN pip install --no-cache-dir virtualenv
#RUN virtualenv /venv
#ENV PATH="/venv/bin:$PATH"
#RUN pip install Flask==2.3.2
#RUN pip install gunicorn==21.2.0
#
#RUN pip install --no-cache-dir .
#RUN pip install gevent
#
#EXPOSE 8080
#ENV TIMEOUT=30
#ENV NUM_WORKERS=4
##CMD ["flask", "--app", "app.py", "run", "--host=0.0.0.0", "--port=8080", "--no-reload"]
#CMD gunicorn --workers $NUM_WORKERS --timeout $TIMEOUT --bind 0.0.0.0:8080 --worker-class gevent app:app

# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements to the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the container
COPY . .

# Expose port 8000
EXPOSE 8008

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]