
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim



# Copy local code to the container image.
WORKDIR /app

COPY requirements.txt .
# Install production dependencies.
RUN pip install -r requirements.txt
#copy all
COPY . .
EXPOSE 8080
ENV FLASK_APP=app.py
# Run the web service on container startup. Here we use the gunicorn

CMD ["gunicorn"  , "--bind", "0.0.0.0:8080", "app:app"]
