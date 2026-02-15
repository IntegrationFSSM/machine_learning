import sys
import os

# Ensure the src module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import run_training_pipeline

if __name__ == "__main__":
    print("Starting Titanic MLOps Pipeline...")
    run_training_pipeline()
    print("Pipeline finished successfully.")
