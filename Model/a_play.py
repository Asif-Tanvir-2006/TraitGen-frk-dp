import a_model
import a_bioclip_VE
import a_gpt2_TD

image_list = [
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg',
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
]
model = a_model.Model(vision_encoder=a_bioclip_VE.BioCLIP, text_decoder=a_gpt2_TD.GPT2Decoder())
print(model.generate(image_list=image_list))
