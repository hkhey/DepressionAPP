# Mental Health Prediction Project
## Overview:
This project aims to develop a robust machine learning model for predicting depression in students based on various mental health-related features. The workflow includes comprehensive data preprocessing, training multiple machine learning models, and creating a Flask application to consume the trained model via a REST API. Additionally, the best-performing model is saved in ONNX format, and its preprocessing transformations are stored in a pickle file.

## Project Structure:
### FastAPI:
The FastAPI directory hosts the FastAPI application responsible for exposing the machine learning model through a REST API.

#### templates: This directory contains HTML templates for the FastAPI application.
#### app.py: The FastAPI application script, defining API endpoints and integrating the machine learning model.
#### Dockerfile: Dockerfile configuration for packaging the FastAPI application.
#### Procfile: Procfile for Heroku deployment.
#### random_forest_model.onnx: The best-performing machine learning model saved in ONNX format.
#### requirements.txt: A text file listing the required Python packages.
#### sklearn_conf_matrix.png: An image file depicting the confusion matrix generated during model evaluation.
### Flask:
The Flask directory contains the Flask application designed to consume the FastAPI-based REST API.

#### mlruns: The directory where MLflow logs and experiment details are stored.
#### templates: This directory holds HTML templates for the Flask application.
#### flask_app.py: The Flask application script, responsible for creating a web interface to interact with the model.
#### Dockerfile: Dockerfile configuration for packaging the Flask application.
#### MLFlow.ipynb: A Jupyter notebook documenting the MLflow experiment, including model training runs and metrics.
#### preprocessing_steps.pkl: The preprocessing steps saved in a pickle file.
#### preprocessing_steps_rf.pkl: Preprocessing steps specifically for the best-performing model.
#### random_forest_model.onnx: The best-performing machine learning model saved in ONNX format.
#### requirements.txt: A text file listing the required Python packages.
#### sklearn_conf_matrix.png: An image file showing the confusion matrix generated during model evaluation.
#### StudentMentalhealth.csv: The dataset used for training the machine learning model.


## How to Run:

### FastAPI:
Navigate to the FastAPI directory. \
Build the Docker image: docker build -t fastapi-app . \
Run the Docker container: docker run -p 8000:8000 fastapi-app

### Flask:
Navigate to the Flask directory. \
Build the Docker image: docker build -t flask-app . \
Run the Docker container: docker run -p 8080:8080 flask-app \

### MLflow Experiment:
The MLflow experiment details, including model training runs, metrics, and artifacts, are documented in the MLFlow.ipynb Jupyter notebook.

### Postman:
The APIs can be tested using Postman. Ensure the correct endpoints are used for both FastAPI and Flask applications.

## Additional Notes:
Ensure Docker is installed for convenient packaging and deployment. \
Experiment with different endpoints and explore the MLflow UI for detailed model tracking. \

## Acknowledgments:
MLflow Documentation: https://www.mlflow.org/ \
FastAPI Documentation: https://fastapi.tiangolo.com/ \
Flask Documentation: https://flask.palletsprojects.com/ \
