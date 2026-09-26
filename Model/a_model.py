import os
import json

import torch
import torch.nn as nn

import a_VE
import a_TD


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Maximum number of TEXT tokens.
# Image tokens and steering-prompt tokens are additional.
MAX_TEXT_LEN = 96

STEERING_PROMPT = "Describe this bird species correctly."

# Species sentence + EOS are more important than ordinary
# attribute tokens.
SPECIES_WEIGHT = 5.0
EOS_WEIGHT = 5.0

GRAD_CLIP_NORM = 1.0


class Model(nn.Module):

    def __init__(
        self,
        vision_encoder,
        text_decoder,
        images_root="/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/"
    ):

        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.ve = a_VE.VisionEncoder(
            vision_encoder
        ).to(DEVICE)


        # ====================================================
        # DECODER
        # ====================================================

        self.td = a_TD.TextDecoder(
            text_decoder.to(DEVICE)
        ).to(DEVICE)


        # ====================================================
        # TOKENIZER
        # ====================================================

        self.tokenizer = self.td.model.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # ====================================================
        # IMAGE -> GPT2 PROJECTION
        # ====================================================

        # Created on first forward pass because we don't
        # necessarily know the encoder dimension beforehand.

        self.image_projection = None


        # ====================================================
        # PATH
        # ====================================================

        self.images_root = images_root


        # ====================================================
        # OPTIMIZER
        # ====================================================

        self.optimizer = None
        self._optimizer_lr = None


    # ========================================================
    # GPT-2 EMBEDDING LAYER
    # ========================================================

    def _get_text_embedding_layer(self):

        return self.td.model.embedding


    # ========================================================
    # CREATE IMAGE PROJECTION
    # ========================================================

    def _ensure_projection(
        self,
        encoder_dim,
        decoder_dim
    ):

        if self.image_projection is None:

            self.image_projection = nn.Linear(
                encoder_dim,
                decoder_dim
            ).to(DEVICE)

            print(
                f"Created image projection: "
                f"{encoder_dim} -> {decoder_dim}"
            )

        return self.image_projection


    # ========================================================
    # TOKENIZE CAPTIONS
    # ========================================================

    def _prepare_captions(self, captions):

        """
        Produces:

            input_ids
            attention_mask
            species_mask

        Every target becomes:

            [attributes]
            [This species is likely ...]
            [EOS]

        Species and EOS are guaranteed to fit.

        species_mask:

            0 -> ordinary attribute token
            1 -> species token / EOS
        """

        eos_id = self.tokenizer.eos_token_id

        all_ids = []
        all_masks = []
        all_species_masks = []

        for caption in captions:

            caption = caption.strip()

            marker = "This species is likely"

            species_pos = caption.find(marker)


            # ------------------------------------------------
            # Split attributes and species sentence
            # ------------------------------------------------

            if species_pos == -1:

                # This should not happen with your generated
                # JSON, but handle it safely.

                attribute_text = caption
                species_text = ""

            else:

                attribute_text = caption[:species_pos].strip()
                species_text = caption[species_pos:].strip()


            # ------------------------------------------------
            # Tokenize separately
            # ------------------------------------------------

            attribute_ids = self.tokenizer(
                attribute_text,
                add_special_tokens=False
            )["input_ids"]

            species_ids = self.tokenizer(
                species_text,
                add_special_tokens=False
            )["input_ids"]


            # ------------------------------------------------
            # Reserve space for:
            #
            # species + EOS
            # ------------------------------------------------

            reserved = len(species_ids) + 1

            max_attribute_tokens = max(
                MAX_TEXT_LEN - reserved,
                0
            )


            # ------------------------------------------------
            # Truncate ONLY attributes
            #
            # Never truncate species.
            # ------------------------------------------------

            attribute_ids = attribute_ids[
                :max_attribute_tokens
            ]


            # ------------------------------------------------
            # Final sequence
            # ------------------------------------------------

            ids = (
                attribute_ids
                + species_ids
                + [eos_id]
            )


            # Safety check

            ids = ids[:MAX_TEXT_LEN]


            # ------------------------------------------------
            # Attention mask
            # ------------------------------------------------

            attention = [
                1
            ] * len(ids)


            # ------------------------------------------------
            # Species mask
            #
            # attributes -> 0
            # species    -> 1
            # EOS        -> 1
            # ------------------------------------------------

            species_mask = (
                [0] * len(attribute_ids)
                + [1] * len(species_ids)
                + [1]
            )

            species_mask = species_mask[
                :MAX_TEXT_LEN
            ]


            all_ids.append(ids)
            all_masks.append(attention)
            all_species_masks.append(species_mask)


        # ====================================================
        # PAD BATCH
        # ====================================================

        batch_size = len(captions)

        max_len = max(
            len(x)
            for x in all_ids
        )

        pad_id = self.tokenizer.pad_token_id


        input_ids = torch.full(
            (
                batch_size,
                max_len
            ),
            pad_id,
            dtype=torch.long,
            device=DEVICE
        )


        attention_mask = torch.zeros(
            (
                batch_size,
                max_len
            ),
            dtype=torch.long,
            device=DEVICE
        )


        species_mask = torch.zeros(
            (
                batch_size,
                max_len
            ),
            dtype=torch.float,
            device=DEVICE
        )


        for i in range(batch_size):

            n = len(all_ids[i])

            input_ids[i, :n] = torch.tensor(
                all_ids[i],
                dtype=torch.long,
                device=DEVICE
            )

            attention_mask[i, :n] = 1

            species_mask[i, :n] = torch.tensor(
                all_species_masks[i],
                dtype=torch.float,
                device=DEVICE
            )


        return (
            input_ids,
            attention_mask,
            species_mask
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        image_paths,
        captions
    ):

        """
        Training sequence:

            [IMAGE TOKENS]
            [STEERING PROMPT]
            [CAPTION TOKENS]
            [EOS]


        Loss:

            IMAGE TOKENS  -> ignored
            PROMPT TOKENS -> ignored
            ATTRIBUTES    -> weight 1
            SPECIES       -> weight 5
            EOS           -> weight 5
        """


        # ====================================================
        # 1. IMAGE ENCODER
        # ====================================================

        image_embeds = self.ve(
            image_paths
        )

        # Expected:

        # [B, N_IMAGE_TOKENS, vision_dim]


        # ====================================================
        # 2. TOKENIZE CAPTIONS
        # ====================================================

        (
            input_ids,
            attn_mask_text,
            species_mask
        ) = self._prepare_captions(
            captions
        )


        # ====================================================
        # 3. CAPTION EMBEDDINGS
        # ====================================================

        text_embedding_layer = (
            self._get_text_embedding_layer()
        )

        text_embeds = text_embedding_layer(
            input_ids
        )

        # [B, T, 768]


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


        prompt_embeds = text_embedding_layer(
            prompt_ids
        )

        # [1, P, 768]


        batch_size = image_embeds.size(0)

        prompt_embeds = prompt_embeds.expand(
            batch_size,
            -1,
            -1
        )

        num_prompt_tokens = (
            prompt_embeds.size(1)
        )


        # ====================================================
        # 5. IMAGE PROJECTION
        # ====================================================

        encoder_dim = image_embeds.shape[-1]
        decoder_dim = text_embeds.shape[-1]

        projection = self._ensure_projection(
            encoder_dim,
            decoder_dim
        )

        image_embeds = projection(
            image_embeds
        )

        num_image_tokens = (
            image_embeds.size(1)
        )


        # ====================================================
        # 6. CONCATENATE
        #
        # [IMAGE] [PROMPT] [TEXT]
        # ====================================================

        combined_embeds = torch.cat(
            [
                image_embeds,
                prompt_embeds,
                text_embeds
            ],
            dim=1
        )


        # ====================================================
        # 7. ATTENTION MASK
        # ====================================================

        image_mask = torch.ones(
            batch_size,
            num_image_tokens,
            dtype=attn_mask_text.dtype,
            device=DEVICE
        )


        prompt_mask = torch.ones(
            batch_size,
            num_prompt_tokens,
            dtype=attn_mask_text.dtype,
            device=DEVICE
        )


        combined_attention_mask = torch.cat(
            [
                image_mask,
                prompt_mask,
                attn_mask_text
            ],
            dim=1
        )


        # ====================================================
        # 8. LABELS
        # ====================================================

        # Image positions:
        #
        # NO LOSS

        image_labels = torch.full(
            (
                batch_size,
                num_image_tokens
            ),
            -100,
            dtype=input_ids.dtype,
            device=DEVICE
        )


        # Prompt positions:
        #
        # NO LOSS

        prompt_labels = torch.full(
            (
                batch_size,
                num_prompt_tokens
            ),
            -100,
            dtype=input_ids.dtype,
            device=DEVICE
        )


        # Caption positions

        text_labels = input_ids.clone()


        # Padding:
        #
        # NO LOSS

        text_labels[
            attn_mask_text == 0
        ] = -100


        # Final labels

        labels = torch.cat(
            [
                image_labels,
                prompt_labels,
                text_labels
            ],
            dim=1
        )


        # ====================================================
        # 9. GPT-2
        # ====================================================

        outputs = self.td.forward(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=None
        )

        logits = outputs.logits


        # ====================================================
        # 10. CAUSAL SHIFT
        # ====================================================

        shift_logits = logits[
            :, :-1, :
        ].contiguous()

        shift_labels = labels[
            :, 1:
        ].contiguous()


        # ====================================================
        # 11. TOKEN-LEVEL CROSS ENTROPY
        # ====================================================

        loss_function = nn.CrossEntropyLoss(
            reduction="none"
        )


        token_loss = loss_function(
            shift_logits.view(
                -1,
                shift_logits.size(-1)
            ),
            shift_labels.view(-1)
        )


        token_loss = token_loss.view(
            shift_labels.shape
        )


        # ====================================================
        # 12. WEIGHT SPECIES + EOS
        # ====================================================

        # species_mask corresponds to labels.
        #
        # Shift it exactly like the labels.

        shift_species_mask = species_mask[
            :, 1:
        ]


        weights = torch.ones_like(
            token_loss
        )


        weights[
            shift_species_mask == 1
        ] = SPECIES_WEIGHT


        # ====================================================
        # 13. IGNORE INVALID POSITIONS
        # ====================================================

        valid = (
            shift_labels != -100
        )


        # ====================================================
        # 14. WEIGHTED LOSS
        # ====================================================

        weighted_loss = (
            token_loss * weights
        )


        loss = (
            weighted_loss[valid].sum()
            /
            weights[valid].sum()
        )


        return loss


    # ========================================================
    # TRAINING
    # ========================================================

    def start_training(
        self,
        epochs=100,
        path=None,
        batch_size=16,
        lr=5e-5,
        mode=True,
        start=0,
        stop=0
    ):

        # ----------------------------------------------------
        # Training mode
        # ----------------------------------------------------

        super().train(True)

        # Vision encoder frozen

        self.ve.eval()


        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        with open(path, "r") as f:

            dataset = json.load(f)


        if stop > start:

            dataset = dataset[
                start:stop + 1
            ]

        else:

            dataset = dataset[
                start:
            ]


        print(
            f"Training samples: {len(dataset)}"
        )


        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(epochs):

            total_loss = 0.0
            total_batches = 0


            for i in range(
                0,
                len(dataset),
                batch_size
            ):

                batch = dataset[
                    i:i + batch_size
                ]


                # --------------------------------------------
                # Image paths
                # --------------------------------------------

                image_paths = [
                    os.path.join(
                        self.images_root,
                        item["imagePath"]
                    )
                    for item in batch
                ]


                # --------------------------------------------
                # Captions
                # --------------------------------------------

                captions = [
                    item["gt"].strip()
                    for item in batch
                ]


                # --------------------------------------------
                # Forward
                #
                # IMPORTANT:
                #
                # This creates image_projection on the first
                # batch.
                # --------------------------------------------

                loss = self(
                    image_paths,
                    captions
                )


                # --------------------------------------------
                # Create optimizer AFTER projection exists
                #
                # This fixes the subtle bug in your original
                # implementation.
                # --------------------------------------------

                if (
                    self.optimizer is None
                    or
                    self._optimizer_lr != lr
                ):

                    self.optimizer = torch.optim.AdamW(
                        filter(
                            lambda p: p.requires_grad,
                            self.parameters()
                        ),
                        lr=lr
                    )

                    self._optimizer_lr = lr

                    print(
                        "Optimizer initialized."
                    )


                # --------------------------------------------
                # Backprop
                # --------------------------------------------

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()


                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    GRAD_CLIP_NORM
                )


                # --------------------------------------------
                # Update
                # --------------------------------------------

                self.optimizer.step()


                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                total_loss += loss.item()

                total_batches += 1


            avg_loss = (
                total_loss
                /
                max(total_batches, 1)
            )


            print(
                f"Epoch "
                f"{epoch + 1}/{epochs} "
                f"| loss = {avg_loss:.4f}"
            )


    # ========================================================
    # GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        image_list,
        max_new_tokens=96,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    ):

        self.eval()


        # ====================================================
        # 1. IMAGE ENCODER
        # ====================================================

        image_embeds = self.ve(
            image_list
        )


        # ====================================================
        # 2. GPT2 DIMENSION
        # ====================================================

        text_embedding_layer = (
            self._get_text_embedding_layer()
        )


        # We need the decoder dimension.

        decoder_dim = (
            text_embedding_layer.weight.shape[1]
        )


        # ====================================================
        # 3. IMAGE PROJECTION
        # ====================================================

        image_embeds = self._ensure_projection(
            image_embeds.shape[-1],
            decoder_dim
        )(
            image_embeds
        )


        # ====================================================
        # 4. STEERING PROMPT
        # ====================================================

        prompt = self.tokenizer(
            STEERING_PROMPT,
            add_special_tokens=False,
            return_tensors="pt"
        )


        prompt_ids = prompt[
            "input_ids"
        ].to(DEVICE)


        prompt_embeds = text_embedding_layer(
            prompt_ids
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


        batch_size = image_embeds.size(0)

        attention_mask = torch.ones(
            batch_size,
            inputs_embeds.size(1),
            dtype=torch.long,
            device=DEVICE
        )


        # ====================================================
        # 6. GENERATE
        # ====================================================

        # Your GPT2Decoder wraps the HuggingFace GPT2 model
        # as self.td.model.gpt2.

        generated = self.td.model.gpt2.generate(

            inputs_embeds=inputs_embeds,

            attention_mask=attention_mask,

            max_new_tokens=max_new_tokens,

            do_sample=do_sample,

            temperature=(
                temperature
                if do_sample
                else None
            ),

            top_p=(
                top_p
                if do_sample
                else None
            ),

            repetition_penalty=repetition_penalty,

            eos_token_id=(
                self.tokenizer.eos_token_id
            ),

            pad_token_id=(
                self.tokenizer.eos_token_id
            )
        )


        # ====================================================
        # 7. DECODE
        # ====================================================

        text = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )


        return text