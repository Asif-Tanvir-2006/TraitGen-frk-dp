import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import a_VE
import a_TD


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_ROOT = "/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0
EPOCHS = 5

class CUBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(IMAGES_ROOT, item["imagePath"])
        return image_path, item["gt"]


class Model(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()

        self.ve = a_VE.VisionEncoder(vision_encoder).to(DEVICE)
        self.td = a_TD.TextDecoder(text_decoder).to(DEVICE)

    def forward(self, image_paths, captions):
        image_features = self.ve(image_paths)
        return self.td(image_features, captions)

    @torch.no_grad()
    def generate(self, image_paths):
        image_features = self.ve(image_paths)
        return self.td.generate(image_features)

    def start_training(self, json_path, start, stop):
        with open(json_path, "r") as f:
            data = json.load(f)
        # data = data[start: stop+1]
        dataset = CUBDataset(data)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda x: (
                [i[0] for i in x],
                [i[1] for i in x]
            )
        )

        self.ve.eval()
        self.td.train()

        first_batch = next(iter(loader))
        self(first_batch[0], first_batch[1])

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE
        )

        for epoch in range(EPOCHS):
            total_loss = 0.0

            for image_paths, captions in loader:
                optimizer.zero_grad()

                loss = self(image_paths, captions)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    GRAD_CLIP_NORM
                )

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if(epoch%100==0):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"loss={avg_loss:.4f}"
                )
        torch.save(self.state_dict(), "/kaggle/working/model.pt")