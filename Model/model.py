import torch
import torch.nn as nn

from Model.VisionEncoder import VisionEncoder
from Model.TextDecoder import GPT2Decoder

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


class Bridge(nn.Module):

    def __init__(self, vision_dim: int, hidden_dim: int):
        super().__init__()

        self.projection = nn.Linear(vision_dim, hidden_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.projection(image_features)