import pickle
from flask import Flask, jsonify, request
import mlflow
import os 


run_id = 'd6b4f9fea5ed447a803f09a0697eae8d'
logged_model = f"runs:/{run_id}/model"

# connect to URI using 
mlflow.set_tracking_uri("http://host.docker.internal:5000")

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# loaded_model.predict({'PUlocationID': 10, 'DOlocationID': 50, 'trip_distance': 10})

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PUlocationID']}_{ride['DOlocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(data):
    y_pred = loaded_model.predict(data)
    return y_pred[0]

app = Flask("duration_predictor")

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=9696)
    
# Run the code in the terminal
# $ python predict.py