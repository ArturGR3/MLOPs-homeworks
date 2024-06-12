import pickle
from flask import Flask, jsonify, request


# load the lin_reg model 
with open('lin_reg.bin', 'rb') as f:
   (dv, model) = pickle.load(f)
   
def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PUlocationID']}_{ride['DOlocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(data):
    X = dv.transform(data)
    y_pred = model.predict(X)
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