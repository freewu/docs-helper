#!/usr/bin/env python
"""
Script to download the all-MiniLM-L6-v2 model to the local model directory
"""
import os
from sentence_transformers import SentenceTransformer

def download_model():
    print("Downloading all-MiniLM-L6-v2 model to local directory...")
    
    # Create model directory if it doesn't exist
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    # Define the local path for the model
    local_model_path = os.path.join(model_dir, "all-MiniLM-L6-v2")
    
    # Check if model already exists
    if os.path.exists(local_model_path):
        print(f"Model already exists at: {local_model_path}")
        print("Skipping download.")
        return
    
    print("Downloading model from Hugging Face Hub...")
    try:
        # Load the model (this will download it to cache)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Save the model to the local directory
        print(f"Saving model to: {local_model_path}")
        model.save(local_model_path)
        
        print("Model downloaded and saved successfully!")
        print(f"Model location: {local_model_path}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_model()