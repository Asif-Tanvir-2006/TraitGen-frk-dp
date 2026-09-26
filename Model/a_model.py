import os
import json

import torch
import torch.nn as nn

import a_VE
import a_TD


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

IMAGES_ROOT = (
    "/kaggle/input/datasets/wenewone/"
    "cub2002011/CUB_200_2011/images/"
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

MAX_TEXT_LEN = 96
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
GRAD_CLIP_NORM = 1.0

# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

STEERING_PROMPT = (
    "Describe this bird species correctly."
)

# ------------------------------------------------------------
# Loss weighting
# ------------------------------------------------------------

NORMAL_TOKEN_WEIGHT = 1.0
SPECIES_TOKEN_WEIGHT = 5.0
EOS_TOKEN_WEIGHT = 5.0

# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

MAX_NEW_TOKENS = 96
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.2


class Model(nn.Module):

    def __init__(
        self,
        vision_encoder,
        text_decoder,
        images_root=IMAGES_ROOT
    ):
        super().__init__()

        self.images_root = images_root

        # ====================================================
        # VISION ENCODER
        # ====================================================

        self.ve = a_VE.VisionEncoder(
            vision_encoder
        ).to(DEVICE)

        # ====================================================
        # TEXT DECODER
        # ====================================================

        self.td = a_TD.TextDecoder(
            text_decoder.to(DEVICE)
        ).to(DEVICE)

        # ====================================================
        # TOKENIZER
        # ====================================================

        self.tokenizer = self.td.model.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # ====================================================
        # IMAGE -> GPT2 PROJECTION
        # ====================================================

        # Created dynamically after seeing the actual
        # encoder and decoder dimensions.

        self.image_projection = None

        # ====================================================
        # OPTIMIZER
        # ====================================================

        self.optimizer = None
        self._optimizer_lr = None

    # ========================================================
    # TEXT EMBEDDING
    # ========================================================

    def _get_text_embedding_layer(self):

        return self.td.model.embedding

    # ========================================================
    # IMAGE PROJECTION
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

        elif (
            self.image_projection.in_features
            != encoder_dim
            or
            self.image_projection.out_features
            != decoder_dim
        ):

            raise RuntimeError(
                "Image projection dimensions changed. "
                f"Existing: "
                f"{self.image_projection.in_features} -> "
                f"{self.image_projection.out_features}, "
                f"new: {encoder_dim} -> {decoder_dim}"
            )

        return self.image_projection

    # ========================================================
    # PREPARE CAPTIONS
    # ========================================================

    def _prepare_captions(
        self,
        captions
    ):
        """
        Creates:

            input_ids
            attention_mask
            species_mask
            eos_mask

        Every target is:

            [attributes]
            [species sentence]
            [EOS]

        The attribute section can be truncated.

        The species sentence and EOS are always preserved.
        """

        eos_id = self.tokenizer.eos_token_id

        all_ids = []
        all_attention_masks = []
        all_species_masks = []
        all_eos_masks = []

        for caption in captions:

            caption = caption.strip()

            species_marker = (
                "This species is likely"
            )

            species_position = caption.find(
                species_marker
            )

            # ------------------------------------------------
            # Split attributes and species
            # ------------------------------------------------

            if species_position == -1:

                attribute_text = caption
                species_text = ""

            else:

                attribute_text = (
                    caption[:species_position]
                    .strip()
                )

                species_text = (
                    caption[species_position:]
                    .strip()
                )

            # ------------------------------------------------
            # Tokenize attributes
            # ------------------------------------------------

            attribute_ids = self.tokenizer(
                attribute_text,
                add_special_tokens=False
            )["input_ids"]

            # ------------------------------------------------
            # Tokenize species sentence
            # ------------------------------------------------

            species_ids = self.tokenizer(
                species_text,
                add_special_tokens=False
            )["input_ids"]

            # ------------------------------------------------
            # Reserve space for:
            #
            # species + EOS
            # ------------------------------------------------

            reserved_tokens = (
                len(species_ids) + 1
            )

            max_attribute_tokens = max(
                MAX_TEXT_LEN - reserved_tokens,
                0
            )

            # ------------------------------------------------
            # Truncate ONLY attributes
            # ------------------------------------------------

            attribute_ids = attribute_ids[
                :max_attribute_tokens
            ]

            # ------------------------------------------------
            # Final target
            # ------------------------------------------------

            ids = (
                attribute_ids
                + species_ids
                + [eos_id]
            )

            # Safety check
            ids = ids[:MAX_TEXT_LEN]

            # ------------------------------------------------
            # Attention
            # ------------------------------------------------

            attention = [
                1
            ] * len(ids)

            # ------------------------------------------------
            # Species mask
            #
            # Attribute = 0
            # Species   = 1
            # EOS       = 0 here
            #
            # EOS gets its own mask below.
            # ------------------------------------------------

            species_mask = (
                [0] * len(attribute_ids)
                + [1] * len(species_ids)
                + [0]
            )

            species_mask = species_mask[
                :MAX_TEXT_LEN
            ]

            # ------------------------------------------------
            # EOS mask
            # ------------------------------------------------

            eos_mask = (
                [0] * (
                    len(attribute_ids)
                    + len(species_ids)
                )
                + [1]
            )

            eos_mask = eos_mask[
                :MAX_TEXT_LEN
            ]

            all_ids.append(ids)
            all_attention_masks.append(
                attention
            )
            all_species_masks.append(
                species_mask
            )
            all_eos_masks.append(
                eos_mask
            )

        # ====================================================
        # PAD BATCH
        # ====================================================

        batch_size = len(captions)

        max_batch_length = max(
            len(ids)
            for ids in all_ids
        )

        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full(
            (
                batch_size,
                max_batch_length
            ),
            pad_id,
            dtype=torch.long,
            device=DEVICE
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                max_batch_length
            ),
            dtype=torch.long,
            device=DEVICE
        )

        species_mask = torch.zeros(
            (
                batch_size,
                max_batch_length
            ),
            dtype=torch.bool,
            device=DEVICE
        )

        eos_mask = torch.zeros(
            (
                batch_size,
                max_batch_length
            ),
            dtype=torch.bool,
            device=DEVICE
        )

        # ====================================================
        # COPY INTO BATCH
        # ====================================================

        for batch_index in range(batch_size):

            sequence_length = len(
                all_ids[batch_index]
            )

            input_ids[
                batch_index,
                :sequence_length
            ] = torch.tensor(
                all_ids[batch_index],
                dtype=torch.long,
                device=DEVICE
            )

            attention_mask[
                batch_index,
                :sequence_length
            ] = 1

            species_mask[
                batch_index,
                :sequence_length
            ] = torch.tensor(
                all_species_masks[batch_index],
                dtype=torch.bool,
                device=DEVICE
            )

            eos_mask[
                batch_index,
                :sequence_length
            ] = torch.tensor(
                all_eos_masks[batch_index],
                dtype=torch.bool,
                device=DEVICE
            )

        return (
            input_ids,
            attention_mask,
            species_mask,
            eos_mask
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
        Sequence:

            [IMAGE TOKENS]
            [STEERING PROMPT]
            [CAPTION TOKENS]
            [EOS]

        Loss:

            image tokens       -> ignored
            prompt tokens      -> ignored
            attribute tokens  -> NORMAL_TOKEN_WEIGHT
            species tokens    -> SPECIES_TOKEN_WEIGHT
            EOS                -> EOS_TOKEN_WEIGHT
            padding            -> ignored
        """

        # ====================================================
        # 1. IMAGE ENCODER
        # ====================================================

        image_embeds = self.ve(
            image_paths
        )

        # Dynamically obtained:
        #
        # [B, N_IMAGE, VISION_DIM]

        batch_size = image_embeds.size(0)
        num_image_tokens = image_embeds.size(1)
        encoder_dim = image_embeds.size(2)

        # ====================================================
        # 2. CAPTIONS
        # ====================================================

        (
            input_ids,
            text_attention_mask,
            species_mask,
            eos_mask
        ) = self._prepare_captions(
            captions
        )

        # ====================================================
        # 3. TEXT EMBEDDINGS
        # ====================================================

        embedding_layer = (
            self._get_text_embedding_layer()
        )

        text_embeds = embedding_layer(
            input_ids
        )

        decoder_dim = text_embeds.size(-1)

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

        # [1, P, decoder_dim]

        num_prompt_tokens = (
            prompt_embeds.size(1)
        )

        # Expand across batch

        prompt_embeds = prompt_embeds.expand(
            batch_size,
            -1,
            -1
        )

        # ====================================================
        # 5. IMAGE PROJECTION
        # ====================================================

        projection = self._ensure_projection(
            encoder_dim,
            decoder_dim
        )

        image_embeds = projection(
            image_embeds
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

        image_attention_mask = torch.ones(
            (
                batch_size,
                num_image_tokens
            ),
            dtype=text_attention_mask.dtype,
            device=DEVICE
        )

        prompt_attention_mask = torch.ones(
            (
                batch_size,
                num_prompt_tokens
            ),
            dtype=text_attention_mask.dtype,
            device=DEVICE
        )

        combined_attention_mask = torch.cat(
            [
                image_attention_mask,
                prompt_attention_mask,
                text_attention_mask
            ],
            dim=1
        )

        # ====================================================
        # 8. LABELS
        # ====================================================

        # Image tokens are never targets.

        image_labels = torch.full(
            (
                batch_size,
                num_image_tokens
            ),
            -100,
            dtype=input_ids.dtype,
            device=DEVICE
        )

        # Prompt tokens are never targets.

        prompt_labels = torch.full(
            (
                batch_size,
                num_prompt_tokens
            ),
            -100,
            dtype=input_ids.dtype,
            device=DEVICE
        )

        # Text tokens are targets.

        text_labels = input_ids.clone()

        # Padding is never a target.

        text_labels[
            text_attention_mask == 0
        ] = -100

        # Full label sequence.

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
        # 11. TOKEN CROSS ENTROPY
        # ====================================================

        loss_function = nn.CrossEntropyLoss(
            reduction="none"
        )

        token_loss = loss_function(
            shift_logits.reshape(
                -1,
                shift_logits.size(-1)
            ),
            shift_labels.reshape(-1)
        )

        token_loss = token_loss.reshape(
            shift_labels.shape
        )

        # ====================================================
        # 12. FULL-SEQUENCE WEIGHT MASK
        # ====================================================

        # IMPORTANT:
        #
        # weights must have exactly the same sequence
        # dimensions as token_loss.
        #
        # [IMAGE] [PROMPT] [TEXT]

        image_weights = torch.full(
            (
                batch_size,
                num_image_tokens
            ),
            NORMAL_TOKEN_WEIGHT,
            dtype=token_loss.dtype,
            device=DEVICE
        )

        prompt_weights = torch.full(
            (
                batch_size,
                num_prompt_tokens
            ),
            NORMAL_TOKEN_WEIGHT,
            dtype=token_loss.dtype,
            device=DEVICE
        )

        text_weights = torch.full(
            text_attention_mask.shape,
            NORMAL_TOKEN_WEIGHT,
            dtype=token_loss.dtype,
            device=DEVICE
        )

        # Species tokens get increased weight.

        text_weights[
            species_mask
        ] = SPECIES_TOKEN_WEIGHT

        # EOS gets its own weight.

        text_weights[
            eos_mask
        ] = EOS_TOKEN_WEIGHT

        # Full sequence.

        weights = torch.cat(
            [
                image_weights,
                prompt_weights,
                text_weights
            ],
            dim=1
        )

        # ====================================================
        # 13. SHIFT WEIGHTS
        # ====================================================

        shift_weights = weights[
            :, 1:
        ]

        # ====================================================
        # 14. VALID POSITIONS
        # ====================================================

        valid = (
            shift_labels != -100
        )

        # ====================================================
        # 15. WEIGHTED LOSS
        # ====================================================

        weighted_loss = (
            token_loss
            * shift_weights
        )

        loss = (
            weighted_loss[valid].sum()
            /
            shift_weights[valid].sum()
        )

        return loss

    # ========================================================
    # TRAIN
    # ========================================================

    def start_training(
        self,
        epochs=1,
        path=None,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        mode=True,
        start=0,
        stop=0
    ):

        super().train(True)

        # Freeze/eval vision encoder.

        self.ve.eval()

        # ====================================================
        # LOAD DATA
        # ====================================================

        with open(path, "r") as f:

            dataset = json.load(f)


            dataset = dataset[
                start:stop + 1
            ]


        print(
            f"Training samples: "
            f"{len(dataset)}"
        )

        # ====================================================
        # TRAINING
        # ====================================================

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

                image_paths = [
                    os.path.join(
                        self.images_root,
                        item["imagePath"]
                    )
                    for item in batch
                ]

                captions = [
                    item["gt"].strip()
                    for item in batch
                ]

                # ------------------------------------------------
                # Forward
                #
                # The first forward creates image_projection.
                # ------------------------------------------------

                loss = self(
                    image_paths,
                    captions
                )

                # ------------------------------------------------
                # Optimizer
                #
                # Created AFTER projection exists.
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Backprop
                # ------------------------------------------------

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                # ------------------------------------------------
                # Gradient clipping
                # ------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    GRAD_CLIP_NORM
                )

                # ------------------------------------------------
                # Update
                # ------------------------------------------------

                self.optimizer.step()

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_loss += loss.item()
                total_batches += 1

            average_loss = (
                total_loss
                /
                max(total_batches, 1)
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| loss = {average_loss:.4f}"
            )

    # ========================================================
    # GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self.model.generate()