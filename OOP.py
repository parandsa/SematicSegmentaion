''' RGB to GrayScale'''

import numpy as np
from skimage.color import rgb2gray

class Pre_Processing_R2G():

    def __init__(self, images):
        self.img = images

    def R_2_G(self):
        Image_Number = self.img.shape[0] #batch number
        Image_Height = self.img.shape[1] #rows number
        Image_Width = self.shape[2]      #columns number


        Image_R_2_G = np.zeros((Image_Number,Image_Height,Image_Width))

        for i in range(Image_Number):
            Image_R_2_G[i] =  rgb2gray(self.img[i])

        return Image_R_2_G
