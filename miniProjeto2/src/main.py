#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 
import numpy as np 
import glob

from PIL import Image


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) 
iteration = 11000


path_gray_image = r'../image_gray/'
path_erosion_image = r'../image_erosion/'
path_dilation_image = r'../image_dilation/'
path_half_gradients_image = r'../image_half_gradients/'
path_origin_image = r'../image_origin/'
extesion_types = '*[.png || .jpg]'


def getNameImage(idx):
    image_extesion = '.png'
    image_name_pattern = '-color-bar'
    image_name = addZero(idx) + str(idx) + image_name_pattern + image_extesion
    
    return image_name

def addZero(idx):
    if idx < 10 and idx > 0 :
        return '0'
    return ''

def loadImageGrayScale(image_origin):
    
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

def halfGradientsOperation(image_gray):
    image_dilation = dilationOperation(image_gray)   
    image_erosion = erosionOperation(image_gray)
    image_result = cv2.subtract(image_erosion, image_dilation) 
    return image_result
    
def saveImageGray(image_gray, idx): 
    image_name = getNameImage(idx)
    image_url = path_gray_image + image_name
    
    cv2.imwrite(image_url, image_gray)
    
def saveImageErosion(image_gray, idx):
    image_name = getNameImage(idx)
    image_url = path_erosion_image + image_name
    
    image_erosion = erosionOperation(image_gray)
    
    cv2.imwrite(image_url, image_erosion)

def saveImageDilation(image_gray, idx): 
    image_name = getNameImage(idx)
    image_url = path_dilation_image + image_name
    
    image_dilation = dilationOperation(image_gray)
    
    cv2.imwrite(image_url, image_dilation)

def saveImageHalfGradients(image_gray, idx):
    image_name = getNameImage(idx)
    image_url = path_half_gradients_image + image_name
    
    image_result = halfGradientsOperation(image_gray)
    cv2.imwrite(image_url, image_result)
    
    
images = [cv2.imread(file) for file in glob.glob( path_origin_image+extesion_types )]
idx = 1
for img in images:
    print(idx)
    
    image_gray = loadImageGrayScale(img)
    saveImageGray(image_gray, idx)
    saveImageDilation(image_gray, idx)
    saveImageErosion(image_gray, idx)
    saveImageHalfGradients(image_gray, idx)
    idx = idx+1