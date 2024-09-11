import torch
from transformers import CLIPProcessor, CLIPModel
import open_clip
from PIL import Image
import numpy as np

# Load the original OpenCLIP model to convert
model_path_openclip = "hf-hub:Marqo/marqo-fashionCLIP"
model_openclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_path_openclip)
tokenizer_openclip = open_clip.get_tokenizer(model_path_openclip)

# Load the converted Hugging Face CLIP model
model_path_hf = "./converted_hf_model"
hf_model = CLIPModel.from_pretrained(model_path_hf)
hf_processor = CLIPProcessor.from_pretrained(model_path_hf)

# Preprocess the image
image_path = "images/polo-shirt.jpg"
image = Image.open(image_path)
image_tensor_openclip = preprocess_val(image).unsqueeze(0)

# Tokenize the text for both models
texts = ["a photo of a red shoe", "a photo of a black shoe"]
text_tensor_openclip = tokenizer_openclip(texts)

# Hugging Face preprocessing
inputs_hf = hf_processor(text=texts, images=image, return_tensors="pt", padding=True)

# Run inference on OpenCLIP model
with torch.no_grad():
    image_features_openclip = model_openclip.encode_image(image_tensor_openclip)
    text_features_openclip = model_openclip.encode_text(text_tensor_openclip)
    image_features_openclip /= image_features_openclip.norm(dim=-1, keepdim=True)
    text_features_openclip /= text_features_openclip.norm(dim=-1, keepdim=True)

# Run inference on Hugging Face model
with torch.no_grad():
    image_features_hf = hf_model.get_image_features(pixel_values=inputs_hf['pixel_values'])
    text_features_hf = hf_model.get_text_features(input_ids=inputs_hf['input_ids'], attention_mask=inputs_hf['attention_mask'])
    image_features_hf /= image_features_hf.norm(dim=-1, keepdim=True)
    text_features_hf /= text_features_hf.norm(dim=-1, keepdim=True)

# Compare the outputs
image_diff = torch.abs(image_features_openclip - image_features_hf).max()
text_diff = torch.abs(text_features_openclip - text_features_hf).max()

print(f"Max difference in image features: {image_diff.item()}")
print(f"Max difference in text features: {text_diff.item()}")

# Compute similarity scores (if you want to compare them)
text_probs_openclip = (100.0 * image_features_openclip @ text_features_openclip.T).softmax(dim=-1)
text_probs_hf = (100.0 * image_features_hf @ text_features_hf.T).softmax(dim=-1)

print("OpenCLIP Label probs:", text_probs_openclip.tolist())
print("HF CLIP Label probs:", text_probs_hf.tolist())