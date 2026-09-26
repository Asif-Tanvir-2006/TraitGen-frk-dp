import torch
import torch.nn as nn

class TextDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        """
            Expects a matrix of batch x number of patches/tokens x decoder_embed_dimensions as input
            \n
            Outpus a matrix of the same dimension. Last token of each batch is what the model generates and is appended to the end of the 
            token sequence.
            \n
        """
        return self.model.forward(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
            Expects 1 x number of patches/image_tokens x decoder_dim as input
            gives text as output
        """
        
        return self.model.generate(*args, **kwargs)
        
