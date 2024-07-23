# End to End ML Deployment Using AutoGluon

This project demonstrates an end-to-end machine learning deployment pipeline covering: 
1) Data acquisition (Kaggle competition using Kaggle API)
2) Data cleaning and data type optimization for memory usage reduction
3) Feature enginering (using automated feature engineering framework OpenFE)
4) Creation of machine learning model using Autogluon (AutoML framework) and experimentation tracking using MlFlow.
5) Model deployment as flask application using gunicorn as web service using Amazon EC2 
6) Visualization using Grafana with Prometheus as a backend 

Deployment of this application in the docker container

7) Unit & Integration test 
8) Make file for orchestration

## Motivation

The goal of this project to create a template of MVP deployment with open source AutoML (Autogluon) with experiment tracking and monitoring. Due to automated nature, it should take minimal effort to adjust this project to a given data-set (by changing target variable and problem type (regression or classification)).

## Context

### Pulling Data from Kaggle

We start by pulling data directly from Kaggle using its API. A convinient wrapper class [kaggle_client](modules/kaggle_client.py) streamlines interactions with Kaggle competitions. It offers functionality for downloading competition data (train, test and sumbition files) and submitting predictions, making it easier to participate in Kaggle competitions programmatically.

Features:
* Automatic Data Management: Creates a data folder and downloads raw competition data (train, test, and submission files) into it.
* Environment Variable Handling: Utilizes a .env file for secure storage of Kaggle credentials.
* Data Download: Fetches and extracts competition data files from Kaggle.
* Submission Handling: Submits prediction files to the competition and retrieves the public score.

### Data Cleaning, Data Type Optimization

Once the raw data is downloaded, we perform several data preprocessing steps using [data_preprocesing](modules/data_preprocesing.py) class. This steps optimizes the data types and renames columns to be able to feed the data to OpenFE pipeline. Below are the main features it does:

* Optimizes data types to reduce memory usage
* Converts object columns to category or datetime
* Downcasts numeric columns (int and float)
* Converts float columns to int when possible
* Adjusts column names for consistency
* Stores processed data and metadata

Based on the sample kaggle competition the size of the data is being reduced by ~70% by applying more optimal data types. 


### OpenFE Feature Generation

We use [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) automated feature generation for tabular data. In a nutshell this framework helps to identify some of the feature interactions or modification that add to predictive power of the model. For the sake of simplicity I choose to pick additional 5 features. A convinient wrapper class [feature_engineering](modules/feature_engineering.py) helps to perform: 
* Stratified sample of data. (To be able to reduce the size of the data such that we can run auto feature generation faster and without loosing a lot of information about distribution of the data)
* Quick AutoGluon model training for feature evaluation
* Automated feature generation using OpenFE
* Data transformation with generated features

***** Here is some analysis on how new features add towards the predictive power of the model. ***

### AutoGluon AutoML MLflow Tracking (Local, Remote)

With our data prepared, we move on to model training using AutoGluon. AutoGluon automates the process of model selection and hyperparameter tuning. We use MLflow to track our experiments, which helps in comparing different models and configurations, both locally and remotely. For our purpose we create a wrapper classs [mlflow_client](modules/mlflow_client.py) that: 

* MLflow server setup and management 
* Experiment creation and tracking 
* AutoGluon model training with various presets
* Model performance logging and artifact storage
* Support for local and remote MLflow tracking servers
* Ability to downsize the model for deployment purposes 


### Kaggle Score

After training, we evaluate our model's performance and submit our predictions to Kaggle to see how our model ranks on the Kaggle leaderboard on unseen data. For this we use [kaggle_client](modules/kaggle_client.py). (TO BE DONE) We also store public score for our MLflow experimentation purposes.

### AutoGluon Deployment as a Web Service

The next step is to deploy our trained model as a web service, making it accessible for real-time predictions. This involves setting up a server, containerizing our model, and deploying it. We also configured Prometheus for tracking latency and number of requests. We structure deployment setup in this folder [web_service_mflow_visualization](web_service_mlflow_visualiztion) with relavant files for deployment. 


### Latency & Accuracy Monitoring Using Grafana & Prometheus

Finally, we set up monitoring for our deployed model using Grafana that source data from Prometheus. Below is sample of Grafana dashboard for latency based on generic data created by [test_multiple_requests](web_service_mlflow_visualiztion/test_multiple_requests.py). 

## Additional

### Integration Tests

We implement integration tests to ensure that our data pipelines, model training, and deployment processes work together seamlessly.

### Unit Tests

Unit tests are written to test individual components of our pipeline, such as data cleaning functions and feature generation scripts, to ensure they work as expected.

---

This README provides a high-level overview of the project. For detailed instructions and code, please refer to the respective files and directories in this repository.