import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

def save_plot(fig, filename, save_dir="report/images"):
    """Saves a matplotlib figure to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"Saved plot to {path}")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="report/images"):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    save_plot(fig, f'confusion_matrix_{model_name.replace(" ", "_")}.png', save_dir)

def plot_roc_curve(y_true, y_prob, model_name, save_dir="report/images"):
    """Generates and saves the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    save_plot(fig, f'roc_curve_{model_name.replace(" ", "_")}.png', save_dir)

def plot_feature_importance(model, feature_names, model_name, save_dir="report/images"):
    """Generates and saves feature importance plot (Tree-based models)."""
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return
        
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    # Take top 20 features if too many
    n_features = min(len(feature_names), 20)
    top_indices = indices[:n_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Feature Importances - {model_name}")
    ax.bar(range(n_features), importances[top_indices], align="center")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([feature_names[i] for i in top_indices], rotation=90)
    ax.set_xlim([-1, n_features])
    plt.tight_layout()
    save_plot(fig, f'feature_importance_{model_name.replace(" ", "_")}.png', save_dir)

def plot_model_comparison(results, save_dir="report/images"):
    """
    Plots a bar chart comparing accuracy (or other metric) across models.
    results: dict {model_name: score}
    """
    names = list(results.keys())
    scores = list(results.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=scores, ax=ax, palette="viridis")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    for i, v in enumerate(scores):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    save_plot(fig, 'model_comparison.png', save_dir)
