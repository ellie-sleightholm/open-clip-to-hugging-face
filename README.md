# Convert Open-Clip to Hugging Face

This repository converts an `open_clip` model into a Hugging Face `transformer` model that can then be used within `transformers` and `sentence-transformers`. 

To perform the conversion yourself, run `main.py`. This will do the following:
* Create a folder called `marqoFashionCLIP` and then download the `open_clip_pytorch_model.bin` file directly from the `marqo-fashionCLIP` Hugging Face model repository.
* Use this `.bin` file to convert the model into Hugging Face architecture
* This new model will be saved in a folder called `converted_hf_model`

## Comparison

Once the new model has been saved in the directory `converted_hf_model`, we can use it for simple examples. 

### Hugging Face
```python
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
# HF CLIP Label probs: [[2.927805567431996e-11, 3.824342442726447e-08, 1.0]]
```

### Sentence Transformer
```python
from sentence_transformers import SentenceTransformer, util, models
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP model
model_path_hf = "./converted_hf_model"

clip = models.CLIPModel(model_path_hf)
model = SentenceTransformer(modules=[clip])

# Encode an image:
img_emb = model.encode(Image.open('images/hat.png'))

# Encode text descriptions
text_emb = model.encode(["a photo of a red shoe", "a photo of a black shoe", "a hat"])

# Compute cosine similarities 
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)
# tensor([[-0.0194,  0.0524,  0.2232]])
```
