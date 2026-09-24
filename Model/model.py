import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.VisionEncoder import VisionEncoder
from Model.TextDecoder import GPT2Decoder

import os


import numpy as np
import matplotlib.pyplot as plt

def visualize_and_save_similarity_heatmap(original_image_tensor, similarity_matrix, prompt_string, save_path):
    """
    Visualizes Cosine Similarity using ONLY PyTorch and Matplotlib.
    Bypasses SciPy, Sklearn, and OpenCV completely to prevent NumPy 2.0 crashes.
    """
    # 1. Select the first batch item and first prompt similarity
    # similarity_matrix shape: (B, num_prompts, num_patches)
    similarity_map = similarity_matrix[0, 0] # (num_patches,)
    
    # Calculate grid size (e.g., 196 patches -> 14x14 grid)
    num_patches = similarity_map.size(0)
    grid_size = int(num_patches ** 0.5)
    
    # Reshape to 2D grid: (grid_size, grid_size)
    similarity_map = similarity_map.view(grid_size, grid_size)

    # Min-max normalize similarity values to 0.0 - 1.0 range
    min_sim, max_sim = similarity_map.min(), similarity_map.max()
    norm_sim_map = (similarity_map - min_sim) / (max_sim - min_sim + 1e-8)

    # 2. Rescale heatmap to match input image dimensions using PyTorch
    B, C, H, W = original_image_tensor.shape
    scaled_sim_map = norm_sim_map.unsqueeze(0).unsqueeze(0) # (1, 1, grid, grid)
    
    # Bilinear interpolation up to full image height & width
    heatmap_resized = F.interpolate(scaled_sim_map, size=(H, W), mode='bilinear', align_corners=False)
    heatmap_np = heatmap_resized.squeeze().cpu().numpy()

    # 3. Convert PyTorch image tensor to NumPy format for plotting
    img_tensor = original_image_tensor[0].cpu().detach()
    # Normalize tensor channels to 0-1 for plotting if they aren't already
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    img_np = img_tensor.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)

    # 4. Plot original image + heatmap overlay using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Render base image
    ax.imshow(img_np)
    
    # Overlay heat map using 'jet' colormap with 50% transparency (alpha=0.5)
    heatmap_overlay = ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
    
    # Add title and colorbar legend
    plt.title(f"Patch Similarity: '{prompt_string}'", fontsize=12, pad=10)
    plt.axis('off')
    fig.colorbar(heatmap_overlay, ax=ax, fraction=0.046, pad=0.04)

    # Save output image
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"[SUCCESS] Heatmap saved without SciPy/OpenCV to: {save_path}")
"""
def visualize_and_save_similarity_heatmap(original_image_tensor, similarity_matrix, prompt_string, save_path):
    # 1. Input Check: Ensure we have a matching BATCH_SIZE
    if original_image_tensor.size(0) != similarity_matrix.size(0):
        raise ValueError(f"Batch size mismatch. Got Images={original_image_tensor.size(0)}, Sims={similarity_matrix.size(0)}")
    
    # 2. Reshape and Normalize the similarity matrix
    # similarity_matrix is (B, num_prompts, num_patches).
    # We select the FIRST prompt (index 0) and the FIRST image in the batch (index 0).
    similarity_map = similarity_matrix[0, 0] # (num_patches,)
    
    # Calculate grid size (e.g., sqrt(196) = 14)
    num_patches = similarity_map.size(0)
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cosine similarity map (size {num_patches}) must be a square number.")

    # Reshape to (H, W) for interpolation
    similarity_map = similarity_map.view(grid_size, grid_size)

    # Convert similarity map to a normalized 0.0 to 1.0 float tensor
    min_sim = similarity_map.min()
    max_sim = similarity_map.max()
    norm_sim_map = (similarity_map - min_sim) / (max_sim - min_sim)

    # 3. Scale and Resize the Heatmap to match the image dimensions
    B, C, H, W = original_image_tensor.shape
    scaled_sim_map = norm_sim_map.unsqueeze(0).unsqueeze(0) # (1, 1, grid, grid)
    resized_heatmap_tensor = F.interpolate(scaled_sim_map, size=(H, W), mode='bilinear', align_corners=False)
    
    # Convert to Numpy (H, W) array, scaled to 0-255 range for OpenCV
    resized_heatmap_np = (resized_heatmap_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 4. Generate the Visualized Blended Image
    # Convert original tensor back to RGB Numpy image (B, C, H, W) -> (H, W, C)
    original_img_np = original_image_tensor[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize image pixel range and convert to BGR (required by OpenCV)
    original_img_np = ((original_img_np - original_img_np.min()) / (original_img_np.max() - original_img_np.min()) * 255).astype(np.uint8)
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    # Create the COLOR heatmap (Map 0-255 to JET colormap: low similarity = blue, high similarity = red)
    heatmap_colored = cv2.applyColorMap(resized_heatmap_np, cv2.COLORMAP_JET)

    # 5. SUPERIMPOSE! (Overlay Heatmap onto the original image)
    # The weight parameters (alpha and beta) control the transparency of the overlay
    alpha = 0.6  # Weight of the heatmap (0.0 to 1.0, higher means darker/more dominant heatmap)
    beta = 1.0 - alpha
    blended_image = cv2.addWeighted(heatmap_colored, alpha, original_img_bgr, beta, 0)

    # Optional: Plot and Save the final output image with title
    # (We save the image directly with OpenCV to avoid Matplotlib dependency if preferred)
    cv2.imwrite(save_path, blended_image)
    
    print(f"[INFO] Successfully generated similarity visualization.")
    print(f"       Prompt used: '{prompt_string}'")
    print(f"       Output saved to: '{save_path}'")
"""

class TraitGen(nn.Module):
    """
    TraitGen: Image Captioning Model

    Components:
        - VisionEncoder: extracts image representations.
        - Bridge: projects visual features into GPT-2 embedding space.
        - GPT2Decoder: generates captions conditioned on image features.
    """

    def __init__(self, args, vision_encoder=None):
        super().__init__()

        self.args = args
        self.vision_encoder = VisionEncoder(args) if vision_encoder is None else vision_encoder
        self.decoder = GPT2Decoder(args)
        self.bridge = Bridge(vision_dim=args.encoder_op_dim, hidden_dim=self.decoder.hidden_dim)

    # ============================================================
    # Prepare inputs for GPT-2 training/validation
    # ============================================================
    def input2decoder(self, prompt_ids, prompt_mask, prefix_embeds, target_ids, target_mask):

        prompt_embeds = self.decoder.gpt2.get_input_embeddings()(prompt_ids)
        target_embeds = self.decoder.gpt2.get_input_embeddings()(target_ids)

        inputs_embeds = torch.cat([prompt_embeds, prefix_embeds, target_embeds], dim=1)

        B, PREFIX_LEN = prompt_ids.size(0), prefix_embeds.size(1)
        device = prompt_ids.device

        prompt_labels = torch.full(prompt_ids.shape, -100, device=device, dtype=torch.long)
        prefix_labels = torch.full((B, PREFIX_LEN), -100, device=device, dtype=torch.long)
        labels = torch.cat([prompt_labels, prefix_labels, target_ids], dim=1)

        prefix_mask = torch.ones((B, PREFIX_LEN), device=device, dtype=torch.long)
        full_mask = torch.cat([prompt_mask, prefix_mask, target_mask], dim=1)

        return inputs_embeds, full_mask, labels


    # ============================================================
    # Training / validation forward pass
    # ============================================================
    def forward(self, image, prompt_ids, prompt_mask, target_ids, target_mask):

        image_features = self.vision_encoder(image).permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)

        inputs_embeds, attention_mask, labels = self.input2decoder(
            prompt_ids, prompt_mask, prefix_embeds, target_ids, target_mask)

        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        return loss

    # ============================================================
    # Caption generation
    # ============================================================

    @torch.no_grad()
    def generate_caption(self, image, prompt_ids, prompt_mask):

        image_features = self.vision_encoder(image).permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)

        prompt_embeds = self.decoder.gpt2.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([prompt_embeds, prefix_embeds], dim=1)

        B, PREFIX_LEN = prompt_ids.size(0), prefix_embeds.size(1)
        prefix_mask = torch.ones((B, PREFIX_LEN), device=prompt_ids.device, dtype=torch.long)
        attention_mask = torch.cat([prompt_mask, prefix_mask], dim=1)

        generated = self.decoder.gpt2.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.92,
            repetition_penalty=1.2, eos_token_id=self.decoder.tokenizer.eos_token_id,
            pad_token_id=self.decoder.tokenizer.eos_token_id,)

        generated_text = self.decoder.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text
    
    @torch.no_grad()
    def generate_image_patches(self,image,prompt_list,device="cuda",base_output_path="./similarity_analysis"):
        image_features = self.vision_encoder(image).permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)
        
        # Step 2: Unfold raw input pixels directly into flattened patch representations (Zero extra weights)
        B, C, H, W = image.shape
        num_patches = prefix_embeds.size(1)
        patch_size = int((H * W / num_patches) ** 0.5)

        reconstructed_patches = image.unfold(2, patch_size, patch_size)\
                                    .unfold(3, patch_size, patch_size)\
                                    .permute(0, 2, 3, 1, 4, 5)\
                                    .contiguous()\
                                    .view(B, num_patches, -1)

        # Step 3: Tokenize text prompts and extract embeddings from pre-trained GPT-2
        tokenized_inputs = self.decoder.tokenizer(
            prompt_list, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        # Retrieve text token embeddings from existing LLM layer
        # Shape: (num_prompts, seq_len, hidden_dim)
        text_token_embeds = self.decoder.gpt2.get_input_embeddings()(tokenized_inputs.input_ids)

        # Masked average to pool token vectors into a single prompt vector
        attention_mask = tokenized_inputs.attention_mask.unsqueeze(-1)  # (num_prompts, seq_len, 1)
        text_embeds = (text_token_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        # Compute cosine similarity between text prompts and image patches
        prefix_embeds_norm = F.normalize(prefix_embeds, p=2, dim=-1)  # (B, num_patches, hidden_dim)
        text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)        # (num_prompts, hidden_dim)

        # Similarity matrix: (B, num_prompts, num_patches)
        similarity_matrix = torch.matmul(text_embeds_norm, prefix_embeds_norm.transpose(-1, -2))
        
        output_dict = {
            "patch_embeddings": prefix_embeds,
            "reconstructed_patches": reconstructed_patches,
            "similarity_matrix": similarity_matrix
        }
        # --- End existing generate_image_patches logic ---


        # ============================================================
        # STEP 4: VISUALIZE SIMILARITYHEATMAP
        # ============================================================
        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        # Iterate through all prompts and generate a visualization for each one
        # Note: original_image_tensor must retain its full spatial dimension (B, C, H, W)
        for p_idx, current_prompt_string in enumerate(prompt_list):
            # Slice the similarity matrix to isolate the matrix for just this prompt: (B, 1, num_patches)
            specific_prompt_sim_matrix = similarity_matrix[:, p_idx:p_idx+1, :]
            
            # Ensure filenames are safe for saving (sanitize/remove spaces if needed)
            filename_prompt = current_prompt_string.replace(' ', '_').replace('"', '')[:30] # Limit filename length
            full_save_path = os.path.join(base_output_path, f"vis_sim_matrix_P{p_idx}_{filename_prompt}.jpg")

            # CALL THE STANDALONE VISUALIZATION FUNCTION from Part 1
            # It handles all interpolation, color-mapping, blending, and saving.
            visualize_and_save_similarity_heatmap(image, specific_prompt_sim_matrix, current_prompt_string, full_save_path)

        return output_dict
        


class Bridge(nn.Module):

    def __init__(self, vision_dim: int, hidden_dim: int):
        super().__init__()

        self.projection = nn.Linear(vision_dim, hidden_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.projection(image_features)

