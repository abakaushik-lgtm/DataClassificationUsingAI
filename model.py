import os
import pickle
import numpy as np

class IrisClassifier:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        self.target_names = ['setosa', 'versicolor', 'virginica']
        self.load_model()
        
    def load_model(self):
        """Load the pickle file containing the trained model."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model successfully loaded from '{self.model_path}'")
            except Exception as e:
                print(f"Error loading model from '{self.model_path}': {e}")
        else:
            print(f"Model file '{self.model_path}' not found. Please train the model first.")

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Predict the species of Iris flower based on 4 features.
        Returns:
            dict: {
                "species": str,
                "confidence": float,
                "probabilities": dict,
                "success": bool,
                "error": str (optional)
            }
        """
        if self.model is None:
            # Try loading again in case it was trained in the meantime
            self.load_model()
            if self.model is None:
                return {
                    "success": False,
                    "error": "Model not loaded. Please ensure the model is trained."
                }
        
        try:
            # Inputs validation
            features = [
                float(sepal_length),
                float(sepal_width),
                float(petal_length),
                float(petal_width)
            ]
            
            # Format for prediction (2D array-like)
            features_array = np.array([features])
            
            # Predict class
            prediction_idx = int(self.model.predict(features_array)[0])
            predicted_species = self.target_names[prediction_idx]
            
            # Get prediction probabilities (confidence)
            probabilities = self.model.predict_proba(features_array)[0]
            confidence = float(probabilities[prediction_idx])
            
            # Create a dictionary of probabilities per species
            prob_dict = {
                self.target_names[i]: round(float(probabilities[i]), 4)
                for i in range(len(self.target_names))
            }
            
            return {
                "success": True,
                "species": predicted_species.capitalize(),
                "confidence": round(confidence, 4),
                "probabilities": prob_dict
            }
            
        except ValueError as ve:
            return {
                "success": False,
                "error": f"Invalid input values. Features must be numbers. Details: {ve}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An error occurred during prediction: {e}"
            }
