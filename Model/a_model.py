import os
import json
import torch
import torch.nn as nn
import a_VE
import a_TD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TEXT_LEN = 96  # max caption length in tokens; adjust if your gts run longer
GRAD_CLIP_NORM = 1.0


class Model(nn.Module):
    def __init__(self, vision_encoder, text_decoder, images_root=""):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ENCODER
        # bioClip = a_bioclip_VE.BioCLIP()
        ve = vision_encoder.to(device)
        self.ve = a_VE.VisionEncoder(ve).to(device)

        # DECODER
        # text_decoder = a_gpt2_TD.GPT2Decoder().to(device)
        td = text_decoder.to(device)
        self.td = a_TD.TextDecoder(td).to(device)

        # Reuse the tokenizer already attached to the underlying GPT2Decoder
        # rather than loading a separate one.
        self.tokenizer = self.td.model.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Bridges BioCLIP's embedding dim to GPT-2's. Created lazily on the
        # first forward call, once we actually see both dimensions.
        self.image_projection = None

        # Prefix to join with the relative "imagePath" values stored in
        # train.json / test.json. Override by passing images_root= here,
        # or by setting model.images_root = "..." later.
        self.images_root = images_root

        # Created lazily inside train(), and kept across calls so momentum
        # (Adam's running averages) carries over between epochs instead of
        # resetting every time train() is called.
        self.optimizer = None
        self._optimizer_lr = None

    def _get_text_embedding_layer(self):
        # GPT2Decoder precomputes this as a plain attribute (not a method),
        # since it's a bare nn.Module, not a HF PreTrainedModel.
        return self.td.model.embedding

    def _ensure_projection(self, encoder_dim, decoder_dim):
        if self.image_projection is None and encoder_dim != decoder_dim:
            self.image_projection = nn.Linear(encoder_dim, decoder_dim).to(DEVICE)
        return self.image_projection

    def forward(self, image_paths, captions):
        """
        Loss computation for one batch. Encodes images, embeds the target
        caption tokens, concatenates them as [image tokens] + [caption
        tokens], and returns the language-model loss computed on the
        caption tokens only (image positions are masked out of the loss
        with label == -100).
        """

        # 1. Encode images -> (B, N_img, encoder_dim). BioCLIP.forward is
        #    hard-wrapped in @torch.no_grad(), so these come in detached --
        #    the encoder itself never gets gradients, only the projection
        #    layer below does (that's expected: it stays frozen).
        image_embeds = self.ve(image_paths)

        # 2. Tokenize captions. Force right-padding for this call only --
        #    the tokenizer defaults to left-padding (used by generate()),
        #    but training needs padding *after* the real tokens so
        #    [image][caption][pad] stays causally correct and lines up
        #    with the -100 label masking below.
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        tokenized = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        self.tokenizer.padding_side = original_padding_side

        input_ids = tokenized["input_ids"]
        attn_mask_text = tokenized["attention_mask"]

        # 3. Embed caption tokens with the decoder's own embedding table
        text_embeds = self._get_text_embedding_layer()(input_ids)

        # 4. Project image embeddings to the decoder's dimension if needed
        encoder_dim = image_embeds.shape[-1]
        decoder_dim = text_embeds.shape[-1]
        projection = self._ensure_projection(encoder_dim, decoder_dim)
        if projection is not None:
            image_embeds = projection(image_embeds)

        # 5. Concatenate: [image tokens] + [caption tokens]
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        batch_size, num_img_tokens, _ = image_embeds.shape
        img_attn_mask = torch.ones(
            batch_size, num_img_tokens, device=DEVICE, dtype=attn_mask_text.dtype
        )
        combined_attn_mask = torch.cat([img_attn_mask, attn_mask_text], dim=1)

        # 6. Labels: ignore image positions and padding, predict caption tokens
        ignore = torch.full(
            (batch_size, num_img_tokens), -100, device=DEVICE, dtype=input_ids.dtype
        )
        text_labels = input_ids.clone()
        text_labels[attn_mask_text == 0] = -100
        labels = torch.cat([ignore, text_labels], dim=1)

        outputs = self.td.forward(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn_mask,
            labels=labels,
        )
        return outputs.loss

    ##Train

    def start_training(self, epochs=100, dataset=None, path=None, batch_size=16, lr=5e-5, mode=True):
        """
        model.train(dataset_slice)               -> train one pass over dataset_slice
        model.train(path="train.json", ...)       -> loads the json, trains one pass over it
        model.train()  /  model.train(False)      -> normal nn.Module mode toggle (unchanged)

        `dataset_slice` is a list of {"imagePath": ..., "gt": ...} dicts --
        e.g. json.load(open("train.json"))[start:end]. Returns the average
        loss over the pass.

        This does exactly one pass over whatever you give it and nothing
        else -- no internal epoch loop, no evaluation, no checkpointing.
        Call it once per epoch yourself, and call it again (wrapped in
        `with torch.no_grad():`) on a test slice to get a validation loss.
        """

        # ---- preserve normal nn.Module.train(mode) behavior ----
        if dataset is None and path is None:
            return super().train(mode if isinstance(mode, bool) else True)

        # ---- otherwise: run one real training pass ----
        super().train(True)   # put everything in training mode...
        self.ve.eval()        # ...except the frozen vision encoder

        if dataset is None:
            with open(path, "r") as f:
                dataset = json.load(f)

        if self.optimizer is None or self._optimizer_lr != lr:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr,
            )
            self._optimizer_lr = lr
        for j in range(epochs):
            print("epochs", j)
            total_loss = 0.0
            total_batches = 0

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                image_paths = [
                    os.path.join(self.images_root, item["imagePath"])
                    for item in batch
                ]
                captions = [item["gt"].strip() for item in batch]

                loss = self(image_paths, captions)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            print(total_loss / max(total_batches, 1))

    ##Generate/Inference
    def generate(self, image_list):
        with torch.no_grad():
            VE_out = self.ve.forward(image_list)
            text = self.td.generate(VE_out)
            # TD_out = td.forward(VE_out)
            # print(VE_out)
        return (text)