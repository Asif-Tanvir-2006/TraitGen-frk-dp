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
        
        
