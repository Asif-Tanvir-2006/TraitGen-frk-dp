import VE
import bioclip_VE



bioClip = bioclip_VE.BioCLIP()
ve = VE.VisionEncoder(bioClip)

image_list =[
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg',
    '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    # '/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg'
]

print(ve.forward(image_list).shape)
