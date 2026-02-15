import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.evaluate import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_model_comparison
import numpy as np
import os

def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

def train_model(model, name, X_train, X_test, y_train, y_test, feature_names=None):
    """Trains a model, logs to MLflow, and generates artifacts."""
    
    with mlflow.start_run(run_name=name):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc, prec, rec, f1 = eval_metrics(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Log params (generic)
        mlflow.log_params(model.get_params())
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Generate and Log Plots (Artifacts)
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, name)
        
        # ROC Curve (needs probabilities)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_prob, name)
        
        # Feature Importance
        if feature_names is not None:
             plot_feature_importance(model, feature_names, name)

        # Upload artifacts to MLflow
        mlflow.log_artifacts("report/images")
        
        return acc

def run_training_pipeline():
    # Setup MLflow
    mlflow.set_experiment("Titanic_Survival_Prediction")
    
    # Load and Preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    results = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    acc_lr = train_model(lr, "Logistic_Regression", X_train, X_test, y_train, y_test, feature_names)
    results["Logistic Regression"] = acc_lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    acc_rf = train_model(rf, "Random_Forest", X_train, X_test, y_train, y_test, feature_names)
    results["Random Forest"] = acc_rf
    
    # 3. SVM (needs probability=True for ROC)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    acc_svm = train_model(svm, "SVM", X_train, X_test, y_train, y_test, feature_names)
    results["SVM"] = acc_svm
    
    # Compare Models
    plot_model_comparison(results)
    print("Training pipeline completed.")

if __name__ == "__main__":
    run_training_pipeline()
