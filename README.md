# Apartment Price Prediction API

A simple REST API for predicting apartment prices using linear regression. Built with FastAPI and scikit-learn.

# Overview

This API predicts apartment prices based on three features:
- Area (square meters)
- Number of rooms
- Building age (years)

The model is trained on synthetic data that reflects real market patterns: larger area and more rooms increase the price, while older buildings decrease it.

## Technologies

- FastAPI - web framework
- scikit-learn - machine learning
- NumPy - numerical operations
- Pydantic - data validation
- Uvicorn - ASGI server
- Gunicorn - production server

Local run (non-Docker)
# 1. install requirements
pip install -r requirements.txt

# 2. start app
uvicorn app:app --reload --port 8000

App will be available at http://localhost:8000

Run by Docker
# 1. build image
docker build -t apartment-price-api .

# 2. start container
docker run -d -p 8000:8000 --name my-api apartment-price-api

Run by Docker Compose
# 1. docker compose up
docker-compose up -d

To stop Docker Compose
docker-compose down
