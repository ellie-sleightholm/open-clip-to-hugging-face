import argparse
import os
import json
import torch

from open_clip import create_model, tokenizer
from transformers import CLIPConfig, CLIPVisionConfig, CLIPTextConfig, CLIPModel, CLIPProcessor, AutoTokenizer


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    assert(hf_attn_layer.num_heads == pt_attn_layer.num_heads)
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    hf_attn_layer.q_proj.weight.copy_(q_proj)
    hf_attn_layer.q_proj.bias.copy_(q_proj_bias)

    hf_attn_layer.k_proj.weight.copy_(k_proj)
    hf_attn_layer.k_proj.bias.copy_(k_proj_bias)

    hf_attn_layer.v_proj.weight.copy_(v_proj)
    hf_attn_layer.v_proj.bias.copy_(v_proj_bias)

    hf_attn_layer.out_proj.weight.copy_(pt_attn_layer.out_proj.weight)
    hf_attn_layer.out_proj.bias.copy_(pt_attn_layer.out_proj.bias)


def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    hf_linear.weight.copy_(pt_linear.weight)
    hf_linear.bias.copy_(pt_linear.bias)


def copy_layer(hf_layer, pt_layer):
    # copy layer norms
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # copy MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # copy attn
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    # copy embeds
    hf_encoder.embeddings.token_embedding.weight.copy_(pt_model.token_embedding.weight)
    hf_encoder.embeddings.position_embedding.weight.copy_(pt_model.positional_embedding)

    # copy layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # copy hidden layers
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.text_projection.weight.copy_(pt_model.text_projection.T)

    # copy text encoder
    copy_encoder(hf_model.text_model, pt_model)


def copy_vison_model_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.visual_projection.weight.copy_(pt_model.visual.proj.T)

    # copy layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # copy embeds
    hf_model.vision_model.embeddings.patch_embedding.weight.copy_(pt_model.visual.conv1.weight)
    hf_model.vision_model.embeddings.class_embedding.copy_(pt_model.visual.class_embedding)
    hf_model.vision_model.embeddings.position_embedding.weight.copy_(pt_model.visual.positional_embedding)

    # copy encoder
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


@torch.no_grad()
def convert_clip_checkpoint(model, pretrained, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design and generate tokenizer and preprocessor config.
    """
    if config_path is not None:
        config = CLIPConfig.from_pretrained(config_path)
    else:
        config = CLIPConfig(
            projection_dim=512,
            text_config_dict=dict(hidden_act='gelu'),
            vision_config_dict=dict(hidden_act='gelu'))
        
        # Example config for B16 / B16 plus
        config = CLIPConfig(
            projection_dim=512,
            text_config_dict=dict(
                hidden_act='gelu',
            ),
            vision_config_dict=dict(
                hidden_act='gelu',
                num_hidden_layers=12,
                patch_size=16
            ))

    print(config)
    hf_model = CLIPModel(config).eval()
    print(hf_model)

    pt_model = create_model(model, pretrained=pretrained, precision='fp32')
    pt_model = pt_model.eval()
    print(pt_model)

    copy_text_model_and_projection(hf_model, pt_model)
    copy_vison_model_and_projection(hf_model, pt_model)
    hf_model.logit_scale = pt_model.logit_scale

    # Prepare dummy inputs for validation
    import numpy as np
    input_ids = torch.tensor([49406] + list(np.arange(1, 77, dtype=int))).unsqueeze(0)
    
    # Normalize pixel_values to be in range [0, 1] and avoid rescaling
    pixel_values = torch.randn(1, 3, 224, 224)
    pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())

    # Validate embeddings
    hf_image_embed = hf_model.get_image_features(pixel_values)
    hf_text_embed = hf_model.get_text_features(input_ids)

    pt_image_embed = pt_model.encode_image(pixel_values)
    pt_text_embed = pt_model.encode_text(input_ids)

    assert torch.allclose(hf_image_embed, pt_image_embed, atol=1e-4)
    assert torch.allclose(hf_text_embed, pt_text_embed, atol=1e-4)

    # Validate logits
    hf_logits_per_image, hf_logits_per_text = hf_model(
        input_ids=input_ids, pixel_values=pixel_values, return_dict=False
    )[:2]

    pt_image_features, pt_text_features, logit_scale = pt_model(pixel_values, input_ids)
    pt_logits_per_image = pt_image_features @ pt_text_features.T * logit_scale
    pt_logits_per_text = pt_logits_per_image.T

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-4)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-4)

    # Save Hugging Face model and weights
    hf_model.save_pretrained(pytorch_dump_folder_path)
    torch.save(pt_model.state_dict(), os.path.join(pytorch_dump_folder_path, 'pytorch_model.bin'))

    # Tokenizer and Preprocessor
    tokenizer_name = "Marqo/marqo-fashionCLIP"
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hf_tokenizer.save_pretrained(pytorch_dump_folder_path)
    
    processor = CLIPProcessor.from_pretrained(tokenizer_name)
    processor.save_pretrained(pytorch_dump_folder_path)

    # Save preprocessor config
    preprocessor_config = {
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "rescale_size": 224,
        "crop_size": 224
    }
    with open(os.path.join(pytorch_dump_folder_path, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_config, f)

    # Validate tokenizer and processor are correct
    test_text = "A photo of a cat"
    tokenized_text = hf_tokenizer(test_text, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    processed_image = processor(images=pixel_values, return_tensors="pt", do_rescale=False).pixel_values

    # Validate that the tokenized text and processed image match expected shapes
    assert tokenized_text["input_ids"].shape == (1, 77), f"Tokenization mismatch: got {tokenized_text['input_ids'].shape}"
    assert processed_image.shape == (1, 3, 224, 224), f"Preprocessing mismatch: got {processed_image.shape}"
    
    print("Tokenizer and preprocessor are correctly configured.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--model", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--pretrained", default=None, type=str, help="Path to fairseq checkpoint")
    args = parser.parse_args()

    convert_clip_checkpoint(args.model, args.pretrained, args.pytorch_dump_folder_path)
