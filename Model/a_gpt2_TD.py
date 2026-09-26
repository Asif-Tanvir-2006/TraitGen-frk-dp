import torch
import torch.nn as nn

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


class GPT2Decoder(nn.Module):
    """
    GPT-2 Decoder with LoRA.

    This module is responsible for:
        - Loading the pretrained GPT-2 language model.
        - Initializing the GPT-2 tokenizer.
        - Configuring padding for batch processing.
        - Applying LoRA for parameter-efficient fine-tuning.
        - Producing language model outputs.
    """

    def __init__(self):
        super().__init__()

        # self.args = args

        base_gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_gpt2.config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.padding_side = "left"

        # LoRA configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "c_attn",
                "c_proj",
                "mlp.c_fc",
                "mlp.c_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.gpt2 = get_peft_model(base_gpt2, peft_config)

        self.hidden_dim = self.gpt2.config.n_embd
        self.embedding = self.gpt2.get_input_embeddings()

    def forward(self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,):

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,)

        return outputs

    @torch.no_grad()
    def generate(
            self,
            image_list,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPETITION_PENALTY
        ):
    
            self.eval()
    
            # ====================================================
            # 1. IMAGE
            # ====================================================
    
            image_embeds = self.ve(
                image_list
            )
    
            batch_size = image_embeds.size(0)
            encoder_dim = image_embeds.size(-1)
    
            # ====================================================
            # 2. GPT2 DIMENSION
            # ====================================================
    
            embedding_layer = (
                self._get_text_embedding_layer()
            )
    
            decoder_dim = (
                embedding_layer.weight.size(1)
            )
    
            # ====================================================
            # 3. IMAGE PROJECTION
            # ====================================================
    
            projection = self._ensure_projection(
                encoder_dim,
                decoder_dim
            )
    
            image_embeds = projection(
                image_embeds
            )
    
            # ====================================================
            # 4. STEERING PROMPT
            # ====================================================
    
            prompt_tokens = self.tokenizer(
                STEERING_PROMPT,
                add_special_tokens=False,
                return_tensors="pt"
            )
    
            prompt_ids = prompt_tokens[
                "input_ids"
            ].to(DEVICE)
    
            prompt_embeds = embedding_layer(
                prompt_ids
            )
    
            prompt_embeds = prompt_embeds.expand(
                batch_size,
                -1,
                -1
            )
    
            # ====================================================
            # 5. INITIAL SEQUENCE
            #
            # [IMAGE] [PROMPT]
            # ====================================================
    
            inputs_embeds = torch.cat(
                [
                    image_embeds,
                    prompt_embeds
                ],
                dim=1
            )
    
            attention_mask = torch.ones(
                (
                    batch_size,
                    inputs_embeds.size(1)
                ),
                dtype=torch.long,
                device=DEVICE
            )
    
            # ====================================================
            # 6. GENERATE
            # ====================================================
    
            # The exact underlying GPT-2 model is used here.
    
            generate_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
    
            # Only provide sampling parameters when sampling
            # is enabled.
    
            if do_sample:
    
                generate_kwargs[
                    "temperature"
                ] = temperature
    
                generate_kwargs[
                    "top_p"
                ] = top_p
    
            generated = self.td.model.gpt2.generate(
                **generate_kwargs
            )
    
            # ====================================================
            # 7. DECODE
            # ====================================================
    
            text = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
    
            return text
    # def generate(self, image_embeddings, max_new_tokens=300):

    #     # image_embeddings: [B, 49, 768]
    #     inputs_embeds = image_embeddings

    #     generated_ids = []

    #     for _ in range(max_new_tokens):

    #         outputs = self.forward(
    #             inputs_embeds=inputs_embeds
    #         )

    #         # logits of the final position
    #         next_token_logits = outputs.logits[:, -1, :]

    #         # simplest possible decoding
    #         next_token_id = torch.argmax(
    #             next_token_logits,
    #             dim=-1
    #         )

    #         generated_ids.append(next_token_id)

    #         # Convert token ID → GPT-2 embedding
    #         next_token_embed = self.gpt2.get_input_embeddings()(
    #             next_token_id
    #         ).unsqueeze(1)

    #         # Append it to the sequence
    #         inputs_embeds = torch.cat(
    #             [inputs_embeds, next_token_embed],
    #             dim=1
    #         )

    #     generated_ids = torch.stack(generated_ids, dim=1)

    #     text = self.tokenizer.batch_decode(
    #             generated_ids,
    #             skip_special_tokens=True
    #         )

    #     return text