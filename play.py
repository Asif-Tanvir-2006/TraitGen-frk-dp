import torch
from accelerate import Accelerator

import Model.model as combined_model
import Model.vision_encoder.bioclip_VE as bioclip_VE
import Model.text_decoder.gpt2_TD as gpt2_TD


def train_fn():
    accelerator = Accelerator()

    model = combined_model.Model(
        vision_encoder=bioclip_VE.BioCLIP(),
        text_decoder=gpt2_TD.GPT2Decoder('openai-community/gpt2')
    )

    # state_dict = torch.load("/kaggle/working/model2.pt", map_location='cpu')
    # model.load_state_dict(state_dict)

    for p in model.ve.parameters():
        p.requires_grad_(False)

    model.start_training('./train.json', accelerator=accelerator)

    return accelerator


def run_inference():
    image_list = [
        # '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0085_564.jpg',
        '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
        # '/kaggle/working/TraitGen-frk-dp/black_spot.jpeg',
        # '/kaggle/working/TraitGen-frk-dp/red_spot.jpeg'
    ]

    model = model.Model(
        vision_encoder=bioclip_VE.BioCLIP(),
        text_decoder=gpt2_TD.GPT2Decoder('openai-community/gpt2')
    )
    model.load_state_dict(torch.load("/kaggle/input/notebooks/skasiftanvir/traitgen2/model2.pt", map_location='cpu'))
    model = model.cuda()

    print(model.generate(image_list))


if __name__ == "__main__":
    accelerator = train_fn()
    accelerator.wait_for_everyone()   # barrier: make sure both ranks finish before anyone proceeds

    # if accelerator.is_main_process:
    # run_inference()