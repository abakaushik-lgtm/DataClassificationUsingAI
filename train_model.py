import pickle
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_model():
    print("Loading Iris dataset...")
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = [name.replace(" (cm)", "").title() for name in iris.feature_names]
    target_names = list(iris.target_names)
    
    # Create DataFrame for statistics
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    # 1. Calculate Dataset Statistics
    dataset_stats = {}
    for col in feature_names:
        dataset_stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "std": round(float(df[col].std()), 2)
        }
    
    # 2. Train-Test Split (80/20 split, stratified to keep balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 3. Train Decision Tree Classifier
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Model Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    report_dict = classification_report(
        y_test, y_pred_test, target_names=target_names, output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    cm_list = cm.tolist() # Convert to list for JSON serialization
    
    # Extract Feature Importances
    importances = model.feature_importances_
    feature_importance_dict = {
        name: round(float(imp), 4) for name, imp in zip(feature_names, importances)
    }
    
    # 5. Save the trained model using pickle
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model successfully saved to '{model_filename}'")
    
    # 6. Save model metadata to model_metadata.json
    metadata = {
        "accuracy": round(test_accuracy, 4),
        "train_accuracy": round(train_accuracy, 4),
        "split_ratio": "80/20",
        "sample_sizes": {
            "total": len(X),
            "train": len(X_train),
            "test": len(X_test)
        },
        "target_names": target_names,
        "feature_names": feature_names,
        "feature_importances": feature_importance_dict,
        "confusion_matrix": cm_list,
        "classification_report": report_dict,
        "dataset_stats": dataset_stats
    }
    
    metadata_filename = 'model_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata successfully saved to '{metadata_filename}'")

if __name__ == '__main__':
    train_and_save_model()
