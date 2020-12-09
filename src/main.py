
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 
import numpy as np 

from PIL import Image
import pytesseract

path_tesseract = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = path_tesseract

kernel = np.ones((11,11), np.uint8) 
iteration = 1000

def addZero(idx):
    if idx < 10:
        return '000'
    elif idx < 100:
        return '00'
    return '0'

def loadImage(idx):
    path = r'../image_plate/'
    image_extesion = '.jpg'
    image_name = 'image_' + addZero(idx) + str(idx) + image_extesion
    
    image_url = path + image_name
    
    image_origin = cv2.imread(image_url)
    
    image_gray = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    
    return image_gray

def erosionOperation(image_gray):
    image_erosion = cv2.erode(image_gray, kernel, iterations=iteration) 
    image_result = cv2.subtract(image_gray, image_erosion)
    return image_result

def dilationOperation(image_gray):
    image_dilation = cv2.dilate(image_gray, kernel, iterations=iteration)
    image_result = cv2.subtract(image_dilation, image_gray) 
    return image_result

def saveImageErosion(image_gray, idx): 
    path_save = r'../image_erosion/'
    image_extesion = '.jpg'
    image_name = 'image_' + addZero(idx) + str(idx) + image_extesion
    image_url = path_save + image_name
    
    image_erosion = erosionOperation(image_gray)
    
    cv2.imwrite(image_url, image_erosion)
    saveValueBoard(path_save,idx)

def saveImageDilation(image_gray, idx): 
    path_save = r'../image_dilation/'
    image_extesion = '.jpg'
    image_name = 'image_' + addZero(idx) + str(idx) + image_extesion
    image_url = path_save + image_name
    
    image_dilation = dilationOperation(image_gray)
    
    cv2.imwrite(image_url, image_dilation)
    


def saveValueBoard(path, idx):
    path_save = path
    image_extesion = '.jpg'
    image_name = 'image_' + addZero(idx) + str(idx) + image_extesion
    image_url = path_save + image_name
    
    
    print(pytesseract.image_to_string(Image.open(image_url)))

for idx in range(1,5,1):
    image_gray = loadImage(idx)
        
    saveImageDilation(image_gray,idx)
    saveImageErosion(image_gray, idx)
    
    