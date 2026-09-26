import torch

import a_VE
import a_TD
import a_bioclip_VE
import a_gpt2_TD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ENCODER
bioClip = a_bioclip_VE.BioCLIP()
ve = a_VE.VisionEncoder(bioClip).to(device)

# DECODER
gpt2 = a_gpt2_TD.GPT2Decoder().to(device)
td = a_TD.TextDecoder(gpt2).to(device)


image_list = [
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg',
    # '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
]


with torch.no_grad():
    VE_out = ve.forward(image_list)
    # TD_out = td.forward(VE_out)
print(VE_out)
print(td.generate(VE_out))