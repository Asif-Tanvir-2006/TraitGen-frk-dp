import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    
    
    def forward(self, *args, **kwargs):
        return self.model.fwd(*args, **kwargs)
        #expects B number of image paths, produces B*P*D matrix, 
        # where P is the number of patches produced per image and 
        # D is the number of dimensions in each patch     
