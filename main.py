import os
from convert_clip import convert_clip_checkpoint
import torch
from transformers import CLIPProcessor, CLIPModel
import open_clip
from PIL import Image
import numpy as np
from comparison import comparison
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

def download_assets(your_converted_model_folder):
    # Create folder for the existing model
    your_existing_model_folder = "marqoFashionCLIP"
    if not os.path.exists(your_existing_model_folder):
        os.makedirs(your_existing_model_folder)
            
    hf_hub_download(repo_id="Marqo/marqo-fashionCLIP", filename="open_clip_pytorch_model.bin", local_dir=your_existing_model_folder)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(your_converted_model_folder):
        os.makedirs(your_converted_model_folder)
            
    hf_hub_download(repo_id="Marqo/marqo-fashionCLIP", filename="tokenizer.json", local_dir=your_converted_model_folder)
    hf_hub_download(repo_id="Marqo/marqo-fashionCLIP", filename="preprocessor_config.json", local_dir=your_converted_model_folder)
    

def main():
    
    # First download assets needed
    download_assets("converted_hf_model")
     
    # Set the parameters for conversion
    model_name = "hf-hub:Marqo/marqo-fashionCLIP"
    pretrained_path = "marqoFashionCLIP/open_clip_pytorch_model.bin"  # Replace with the path to your pretrained weights
    pytorch_dump_folder_path = "./converted_hf_model"  # Replace with the desired output folder for the converted model
    
    # Call the conversion function
    convert_clip_checkpoint(model_name, pretrained_path, pytorch_dump_folder_path)
    
    # Compute comparison between models
    comparison()

if __name__ == "__main__":
    main()