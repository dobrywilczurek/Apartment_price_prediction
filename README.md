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
