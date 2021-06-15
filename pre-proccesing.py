'''Pre-Processing Code Includes:
1- One-Hot Encoding
2- Image Resizing
3- Image Agmuntaion
4- Image Re-color
5- Image Saving
6- Show Result'''


#1- Import Required Module 
import cv2
import os
import sys
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from skimage.viewer import ImageViewer
from OOP import Pre_Processing_R2G

#2- Set Image Path
IMAGE_PATH = './data/imgs/'
MASK_PATH = './data/masks/'

#3- Get Images and Set Some Parameters - One-Hot Encoding Technique
IMG_HEIGHT = 256 
IMG_WIDTH =  256
IMG_CHANALS = 3

IMG_DATASET = next(os.walk(IMAGE_PATH))[2]
print(IMG_DATASET)

Inputs = np.zeros((len(IMG_DATASET), IMG_HEIGHT, IMG_WIDTH, IMG_CHANALS), dtype= np.uint8)
Grund_Truth = np.zeros((len(IMG_DATASET), IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.bool)

print(Inputs.shape)
print(Grund_Truth.shape)

print('Loading Images & Masks , please wait')

sys.stdout.flush()

for n,f in tqdm(enumerate(IMG_DATASET), total=len(IMG_DATASET)):
    Images = imread(IMAGE_PATH + f )[:,:,:IMG_CHANALS]
    Inputs[n] = Images

    Masks = cv2.imread(MASK_PATH + f)
    Masks = np.squeeze(Masks).astype(np.bool)


    Grund_Truth[n,:,:,0]=~Masks
    Grund_Truth[n,:,:,1]=Masks


print('Loading Images and Masks Completed Successfully')


#4- Define Funciton : Re_Size , Augmentation , Re-Color , Im_Saving

IMG_HEIGHT_RESIZED = 100 
IMG_WIDTH_RESIZED =  100
IMG_CHANALS_RESIZED = 3

 
#4.1 Function_1 : Pre_Process_Augmentation():
IMAGE_INITIAL_PATH = './data/agumentaion/imgs/'
IMAGE_AGUMENTAION_PATH ='./data/rotated/'
Data_Gen = image.ImageDataGenerator(rotation_range=30)
IMG_AUG = Data_Gen.flow_from_directory(IMAGE_INITIAL_PATH,IMAGE_AGUMENTAION_PATH, batch_size=1,save_to_dir=IMAGE_AGUMENTAION_PATH,save_prefix='Aug')

for i in range(6):
    IMG_AUG.next()



def Pre_Process_Augmentation(Path_Images):
    Image_list = glob.glob(Path_Images)
    Figure = plt.figure()

    for i in range(6):
        Images_A = Image.open(Image_list[i])
        Sub_Image_Show=Figure.add_subplot(231+i)
        Sub_Image_Show.imshow(Images_A)
        plt.show()
    return Figure


Image_original = Pre_Process_Augmentation(IMAGE_INITIAL_PATH + 'imgs/*')
Image_original.savefig(IMAGE_AGUMENTAION_PATH + '/original.png', dpi = 200, papertype='a5')


IMAGE_AGUMENTAION = Pre_Process_Augmentation(IMAGE_AGUMENTAION_PATH + '/*')
IMAGE_AGUMENTAION.savefig(IMAGE_AGUMENTAION_PATH + '/rotate.png', dpi = 200, papertype='a5')


#4.2 Function_1 : Pre_Process_Re_Size():
def Pre_Process_Re_Size(imgs):
    img_p = np.zeros((imgs.shape[0], IMG_HEIGHT_RESIZED, IMG_WIDTH_RESIZED, IMG_CHANALS_RESIZED), dtype = np.unit8)
    for i in range(imgs.shape[0]):
        img_p[i] = resize(imgs[i],(IMG_HEIGHT_RESIZED,IMG_WIDTH_RESIZED,IMG_CHANALS_RESIZED), preserve_range=True)
        return img_p
 
 Image_Resize = Pre_Process_Re_Size(Inputs)

#4.3 Function_1 : Pre_Process_ Re-Color():

Object_Pre_Process = Pre_Processing_R2G(Inputs)
Gray_Scale =  Pre_Processing_R2G.R_2_G(Object_Pre_Process)


#4.4 Function_1 : Pre_Process_Im_Saving():
New = './data/GrayImages/'
def Pre_Process_Im_Saving(path_images, path_output, tensor):
    for i,filename in enumerate(os.listdir(path_images)):
        imsave(fname='{}{}'.format(path_output,filename), arr=tensor[i])
        print('{}'.format(i,filename))


Pre_Process_Im_Saving(IMAGE_PATH,New,Gray_Scale)


#5. Show Result

ix = random.randint(0,len(Inputs))

img=Inputs[ix]
mask=Grund_Truth[ix]
resize=Image_Resize[ix]
gray =Gray_Scale[ix]


print('Input Image')
imshow(img)
plt.show()


print('Mask)
imshow(mask[:,:,0])
plt.show()


print('resized')
imshow(resize)
plt.show()


print('gray')
imshow(gray)
plt.show()





image_view = ImageViewer(gray)
imge_view.show()
