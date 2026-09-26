import a_model
import a_bioclip_VE
import a_gpt2_TD

image_list = [
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg'
]
model = a_model.Model(vision_encoder=a_bioclip_VE.BioCLIP(), text_decoder=a_gpt2_TD.GPT2Decoder('openai-community/gpt2'))
model.start_training(100, './train3.json', 16, 1e-4, True, 0,0)
print(model.generate(image_list=image_list))
