import pandas as pd
import pickle
import sys
import os
from autogluon.tabular import TabularPredictor, TabularDataset
from flask import Flask, request, jsonify


save_path = "autogloun_for_deployment"
predictor = TabularPredictor.load("test_model_ag")
predictor.refit_full()
predictor_clone = predictor.clone_for_deployment(path=save_path, return_clone=True)

size_original = predictor.disk_usage()
size_opt = predictor_clone.disk_usage()
print(f"Size Original:  {size_original} bytes")
print(f"Size Optimized: {size_opt} bytes")
print(
    f"Optimized predictor achieved a {round((1 - (size_opt/size_original)) * 100, 1)}% reduction in disk usage."
)


test_example = {
    "id": 0,
    "store_sales_in_millions_": 8.609999656677246,
    "unit_sales_in_millions_": 3,
    "total_children": 2,
    "num_children_at_home": 2,
    "avg_cars_at_home_approx__1": 2,
    "gross_weight": 10.300000190734863,
    "recyclable_package": 1,
    "low_fat": 0,
    "units_per_case": 32,
    "store_sqft": 36509,
    "coffee_bar": 0,
    "video_store": 0,
    "salad_bar": 0,
    "prepared_food": 0,
    "florist": 0,
    "cost": 62.09000015258789,
    "autoFE_f_0": 6.140243153628883e-05,
    "autoFE_f_1": 0.0,
    "autoFE_f_2": 0.9808552748753283,
    "autoFE_f_3": 36511.0,
    "autoFE_f_4": 2992.0,
}
test_data = pd.DataFrame([test_example])

model_1.predict(test_data)

# Load the model
try:
    predictor = TabularPredictor.load("test_model_ag/models/WeightedEnsemble_L2/model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = Flask("prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        data_df = TabularDataset([data])
        pred = predictor.predict(data_df)[0]
        result = {"cost": round(float(pred), 2)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
