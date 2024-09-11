import torch
from transformers import CLIPProcessor, CLIPModel
import open_clip
from PIL import Image
import numpy as np

# Load the original OpenCLIP model to convert ()
model_path_openclip = "hf-hub:Marqo/marqo-fashionCLIP"
model_openclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_path_openclip)
tokenizer_openclip = open_clip.get_tokenizer(model_path_openclip)

# Preprocess the image
image_path = "images/hat.png"
image = Image.open(image_path)
image_tensor_openclip = preprocess_val(image).unsqueeze(0)

# Tokenize the text for both models
texts = ["a photo of a red shoe", "a photo of a black shoe", "a hat"]
text_tensor_openclip = tokenizer_openclip(texts)

# Run inference on OpenCLIP model
with torch.no_grad():
    image_features_openclip = model_openclip.encode_image(image_tensor_openclip)
    text_features_openclip = model_openclip.encode_text(text_tensor_openclip)
    image_features_openclip /= image_features_openclip.norm(dim=-1, keepdim=True)
    text_features_openclip /= text_features_openclip.norm(dim=-1, keepdim=True)

# Compute similarity scores
text_probs_openclip = (100.0 * image_features_openclip @ text_features_openclip.T).softmax(dim=-1)

print("OpenCLIP Label probs:", text_probs_openclip.tolist())