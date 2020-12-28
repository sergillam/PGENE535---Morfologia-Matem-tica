import cv2
import glob
import numpy as np
import sys
import os
from PIL import Image
import time

d = 0
cpt = 0
i = 0
path = r'../SAVE'
#img_dir = "/home/luiz/PycharmProjects/open"

# images = [cv2.imread(file) for file in glob.glob("/home/luiz/PycharmProjects/open/*.png")]
images = [cv2.imread(file) for file in glob.glob('../image_origin/*[.png || .jpg]')]
# aqui colocase o caminho da base de dados salvo na sua maquina

for img in images:
    i = i +1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('pic' + str(cpt) + '.png ', gray)

    cpt += 1


    img_erosion = gray.copy()
    img_dilate = gray.copy()
    kernel = np.ones((3, 3), np.uint8)
    #imagem erodida
    erosion = cv2.erode(img_erosion, kernel, iterations=5)
    cv2.imwrite(os.path.join(path, 'imgERODIDA' + str(cpt) + '.png '), erosion)
    #cv2.imshow('Erosion_Image', erosion)

    # Imagem dilatada
    dilation = cv2.dilate(img_dilate, kernel, iterations=5)
    cv2.imwrite(os.path.join(path, 'imgDILATADA' + str(cpt) + '.png '), dilation)
    #cv2.imshow('Erosion_Dilate', dilation)

    # OPERAÇAO P+(F) = DILATE - F
    img_p_DILATE = cv2.subtract(dilation, gray)
    #cv2.imshow('OPERAtionP= DILATE - (f) ', img_p_DILATE)
    #cv2.imwrite('img_p_DILATE' + str(cpt) + '.png ', img_p_DILATE)
    cv2.imwrite(os.path.join(path,'img_p_DILATE' + str(cpt) + '.png '), img_p_DILATE)


    # OPERAÇAO P-(F) = F - EROSION
    img_p_EROSION = cv2.subtract(gray, erosion)
    #cv2.imshow('OPERAtionP= = (f) - EROSION ',  img_p_EROSION)
    cv2.imwrite(os.path.join(path, 'img_p_EROSION' + str(cpt) + '.png '), img_p_EROSION)


    # defining the gradient function
    # over the image and structuring element
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(os.path.join(path, 'img_GRADIENT' + str(cpt) + '.png '), gradient)
    #cv2.imshow('Image_GRADIENT', gradient)
    print("Imagem", +  i)



    cv2.waitKey(0)  # Wait for user interaction
    cv2.destroyAllWindows()