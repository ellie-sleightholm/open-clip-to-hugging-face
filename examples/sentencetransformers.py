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
