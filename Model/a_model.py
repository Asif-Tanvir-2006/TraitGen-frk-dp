import torch
import torch.nn as nn
import a_VE
import a_TD





class Model(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
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





##Train







    ##Generate/Inference
    def generate(self, image_list):
        with torch.no_grad():
            VE_out = self.ve.forward(image_list)
            text = self.td.generate()
            # TD_out = td.forward(VE_out)
        # print(VE_out)
        return (text)