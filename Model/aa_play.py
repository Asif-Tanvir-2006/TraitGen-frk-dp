import torch
from accelerate import Accelerator

import a_model
import a_bioclip_VE
import a_gpt2_TD


def train_fn():
    accelerator = Accelerator()

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2')
    )

    state_dict = torch.load("/kaggle/working/model2.pt", map_location='cpu')
    model.load_state_dict(state_dict)

    for p in model.ve.parameters():
        p.requires_grad_(False)

    model.start_training('./train3.json', accelerator=accelerator)

    return accelerator


def run_inference():
    image_list = [
        # '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0085_564.jpg',
        '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg'
    ]

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2')
    )
    model.load_state_dict(torch.load("/kaggle/input/notebooks/skasiftanvir/traitgen2/model2.pt", map_location='cpu'))
    model = model.cuda()

    print(model.generate(image_list))


if __name__ == "__main__":
    # accelerator = train_fn()
    # accelerator.wait_for_everyone()   # barrier: make sure both ranks finish before anyone proceeds

    # if accelerator.is_main_process:
    run_inference()