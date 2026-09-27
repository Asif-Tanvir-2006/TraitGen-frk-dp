import a_model
import a_bioclip_VE
import a_gpt2_TD
import torch
image_list = [
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0085_564.jpg'
]
model = a_model.Model(vision_encoder=a_bioclip_VE.BioCLIP(), text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2'))
model.load_state_dict(
    torch.load(
        "/kaggle/working/model.pt",
        map_location='cuda'
    )
)
model.start_training('./train3.json')

print(model.generate(image_list))
