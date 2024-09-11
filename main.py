import os
from convert_clip import convert_clip_checkpoint
import torch
from transformers import CLIPProcessor, CLIPModel
import open_clip
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

def download_assets(folder_name, model_name, weights_filename):
    """
    Downloads the necessary assets for the model conversion.
    Specifically, it creates a folder if it doesn't exist, 
    and downloads the 'open_clip_pytorch_model.bin' file from the Hugging Face Hub.
    """    
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Download the 'open_clip_pytorch_model.bin' file into the model folder
    hf_hub_download(repo_id=model_name, filename=weights_filename, local_dir=folder_name)

def main():
    """
    Main function that orchestrates the downloading of model assets and converts
    the OpenCLIP model into the Hugging Face format using a predefined conversion function.
    """
    # Step 1: Download 'open_clip_pytorch_model.bin' from your models Hugging Face model repo
    
    # Model name for the original OpenCLIP checkpoint
    model_name = "Marqo/marqo-fashionCLIP"   # Replace with your model name on Hugging Face
    hf_model_name = f"hf-hub:{model_name}"   # This points to the Hugging Face hub
    
    # Pretrained weights
    weights_filename = "open_clip_pytorch_model.bin"   # The name of your pretrained weights for your model
    
    # Name of folder to store open-clip weights file locally
    folder_name = "marqoFashionCLIP" 
    
    # Download the .bin file 
    download_assets(folder_name, model_name, weights_filename)
    
    # Step 2: Set the parameters for the conversion
    
    # Path to the downloaded pretrained OpenCLIP model weights
    pretrained_path = f"{folder_name}/{weights_filename}"  # Replace with the path to your pretrained weights
    
    # Path to save the converted Hugging Face CLIP model
    pytorch_dump_folder_path = "./converted_hf_model"  # Replace with the desired output folder for the converted model
    
    # Step 3: Call the conversion function from the `convert_clip` module
    convert_clip_checkpoint(hf_model_name, pretrained_path, pytorch_dump_folder_path)

if __name__ == "__main__":
    main()
