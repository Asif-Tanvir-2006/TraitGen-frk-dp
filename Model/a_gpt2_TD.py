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

        base_gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-small")
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-small")

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

