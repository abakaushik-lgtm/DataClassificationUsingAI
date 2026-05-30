import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import IrisClassifier

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for API integrations

# Initialize the classifier wrapper
classifier = IrisClassifier(model_path='model.pkl')

METADATA_PATH = 'model_metadata.json'

@app.route('/')
def home():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict Iris flower species based on features.
    Expected JSON payload:
    {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({
            "success": False,
            "error": "Missing request body. Expected JSON data."
        }), 400
        
    required_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing_features = [feat for feat in required_features if feat not in data]
    
    if missing_features:
        return jsonify({
            "success": False,
            "error": f"Missing required measurements: {', '.join(missing_features)}"
        }), 400
        
    # Extract features and validate numbers
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
        
        # Ranges check for physical plausibility (basic validation)
        for name, val in [("Sepal Length", sepal_length), ("Sepal Width", sepal_width), 
                          ("Petal Length", petal_length), ("Petal Width", petal_width)]:
            if val <= 0:
                return jsonify({
                    "success": False,
                    "error": f"{name} must be a positive number greater than 0."
                }), 400
            if val > 15.0: # Safeguard for outlier input
                return jsonify({
                    "success": False,
                    "error": f"{name} ({val} cm) seems physically unrealistic for an Iris flower (limit is 15 cm)."
                }), 400
                
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Invalid measurements provided. All values must be numerical."
        }), 400

    # Call classifier to make predictions
    prediction_result = classifier.predict(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width
    )
    
    if not prediction_result["success"]:
        return jsonify(prediction_result), 500
        
    return jsonify(prediction_result)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Retrieve classification metrics, dataset statistics, feature importances, 
    and confusion matrix details from the trained model's metadata.
    """
    if not os.path.exists(METADATA_PATH):
        return jsonify({
            "success": False,
            "error": "Model metadata file not found. Ensure train_model.py has been run."
        }), 404
        
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return jsonify({
            "success": True,
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to load model metadata: {e}"
        }), 500

if __name__ == '__main__':
    # Run development server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
