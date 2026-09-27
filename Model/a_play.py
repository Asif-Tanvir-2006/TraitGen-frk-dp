# import a_model
# import a_bioclip_VE
# import a_gpt2_TD
# import torch
# import torch.nn as nn
# image_list = [
#     '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0085_564.jpg'
# ]
# # model = a_model.Model(vision_encoder=a_bioclip_VE.BioCLIP(), text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2'))
# # model = nn.DataParallel(model)
# # model = model.cuda()
# # model.load_state_dict(
# #     torch.load(
# #         "/kaggle/working/model.pt",
# #         map_location='cuda'
# #     )
# # )
# # model.start_training('./train3.json')

# # print(model.generate(image_list))
# model = a_model.Model(
#     vision_encoder=a_bioclip_VE.BioCLIP(),
#     text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2')
# )

# # load into the raw model first
# model.load_state_dict(torch.load("/kaggle/working/model.pt", map_location='cpu'))

# model = model.cuda()
# model = nn.DataParallel(model)

# model.module.start_training('./train3.json')
# print(model.module.generate(image_list))



import json
import torch
from accelerate import Accelerator, notebook_launcher

import a_model
import a_bioclip_VE
import a_gpt2_TD


def train_fn():
    accelerator = Accelerator()

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2')
    )

    state_dict = torch.load("/kaggle/working/model.pt", map_location='cpu')
    model.load_state_dict(state_dict)

    # BioCLIP forward runs under @torch.no_grad(), so it never produces
    # gradients — freeze it explicitly or DDP will error expecting them
    for p in model.ve.parameters():
        p.requires_grad_(False)

    model.start_training('./train3.json', accelerator=accelerator)


notebook_launcher(train_fn, num_processes=2)

# --- inference: single GPU, back in the main process, after training ---
image_list = [
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0085_564.jpg'
]

model = a_model.Model(
    vision_encoder=a_bioclip_VE.BioCLIP(),
    text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2')
)
model.load_state_dict(torch.load("/kaggle/working/model2.pt", map_location='cpu'))
model = model.cuda()

print(model.generate(image_list))