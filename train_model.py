import pickle
import json
import os
import numpy as np
import pandas as pd

# Set matplotlib backend to Agg to run in headless mode without GUI window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def train_and_save_model():
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = [name.replace(" (cm)", "").title() for name in iris.feature_names]
    target_names = list(iris.target_names)
    
    # Calculate Dataset Statistics
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    dataset_stats = {}
    for col in feature_names:
        dataset_stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "std": round(float(df[col].std()), 2)
        }
    
    # Train-Test Split (80/20 split, stratified to keep balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # --- 1. Train Decision Tree Classifier ---
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_y_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)
    
    # --- 2. Train Random Forest Classifier ---
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    
    # --- 3. Train Logistic Regression Classifier ---
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_y_pred)
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # Let's save the highest-performing model for predictions in the app!
    best_model = rf_model
    best_accuracy = rf_accuracy
    best_model_name = "Random Forest"
    
    if dt_accuracy > rf_accuracy and dt_accuracy > lr_accuracy:
        best_model = dt_model
        best_accuracy = dt_accuracy
        best_model_name = "Decision Tree"
    elif lr_accuracy > rf_accuracy and lr_accuracy > dt_accuracy:
        best_model = lr_model
        best_accuracy = lr_accuracy
        best_model_name = "Logistic Regression"
        
    print(f"Saving the best model ({best_model_name}) with accuracy {best_accuracy:.4f} to 'model.pkl'...")
    
    # Save the selected best model using pickle
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
        
    # Generate classification report for the best model
    best_pred_test = best_model.predict(X_test)
    best_pred_train = best_model.predict(X_train)
    report_dict = classification_report(
        y_test, best_pred_test, target_names=target_names, output_dict=True
    )
    
    # Generate confusion matrix for the best model
    cm = confusion_matrix(y_test, best_pred_test)
    cm_list = cm.tolist()
    
    # --- Plot & Save Confusion Matrix Image using scikit-learn & matplotlib ---
    print("Generating Confusion Matrix plot image...")
    # Capitalize target names for nice display labels
    capitalized_targets = [name.capitalize() for name in target_names]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=capitalized_targets)
    
    # Render with nice styled dimensions and custom color palette
    fig, ax = plt.subplots(figsize=(6.5, 6))
    disp.plot(cmap=plt.cm.Purples, ax=ax, colorbar=False)
    
    plt.title(f"Confusion Matrix ({best_model_name})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Species", fontsize=11, labelpad=10)
    plt.ylabel("Actual Species", fontsize=11, labelpad=10)
    plt.tight_layout()
    
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Save image
    confusion_matrix_img_path = 'static/confusion_matrix.png'
    plt.savefig(confusion_matrix_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated and saved '{confusion_matrix_img_path}'")
    
    # Extract Feature Importances
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = rf_model.feature_importances_
        
    feature_importance_dict = {
        name: round(float(imp), 4) for name, imp in zip(feature_names, importances)
    }
    
    # Save comparison data and metadata to model_metadata.json
    metadata = {
        "accuracy": round(best_accuracy, 4),
        "train_accuracy": round(accuracy_score(y_train, best_pred_train), 4),
        "algorithm": best_model_name,
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
        "dataset_stats": dataset_stats,
        "model_comparison": [
            {"model": "Decision Tree", "accuracy": round(dt_accuracy, 4)},
            {"model": "Random Forest", "accuracy": round(rf_accuracy, 4)},
            {"model": "Logistic Regression", "accuracy": round(lr_accuracy, 4)}
        ]
    }
    
    metadata_filename = 'model_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata successfully saved to '{metadata_filename}'")

if __name__ == '__main__':
    train_and_save_model()
