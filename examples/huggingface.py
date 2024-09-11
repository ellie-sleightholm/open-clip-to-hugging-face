import torch
from transformers import CLIPProcessor, CLIPModel
import open_clip
from PIL import Image
import numpy as np

# Load the converted Hugging Face CLIP model
model_path_hf = "./converted_hf_model"
hf_model = CLIPModel.from_pretrained(model_path_hf)
hf_processor = CLIPProcessor.from_pretrained(model_path_hf)

# Preprocess the image
image_path = "images/hat.png"
image = Image.open(image_path)

# Tokenize the text for both models
texts = ["a photo of a red shoe", "a photo of a black shoe", "a hat"]

# Hugging Face preprocessing
inputs_hf = hf_processor(text=texts, images=image, return_tensors="pt", padding=True)

# Run inference on Hugging Face model
with torch.no_grad():
    image_features_hf = hf_model.get_image_features(pixel_values=inputs_hf['pixel_values'])
    text_features_hf = hf_model.get_text_features(input_ids=inputs_hf['input_ids'], attention_mask=inputs_hf['attention_mask'])
    image_features_hf /= image_features_hf.norm(dim=-1, keepdim=True)
    text_features_hf /= text_features_hf.norm(dim=-1, keepdim=True)

# Compute similarity scores
text_probs_hf = (100.0 * image_features_hf @ text_features_hf.T).softmax(dim=-1)

print("HF CLIP Label probs:", text_probs_hf.tolist())