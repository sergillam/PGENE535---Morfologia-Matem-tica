
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 
import numpy as np 
import requests

from PIL import Image
import pytesseract


path_tesseract = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = path_tesseract

kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   (3,3))#(51,51))  
iteration = 20000
amount_images = 127

path_gray_image = r'../image_gray/'
path_white_hat_image = r'../image_white_hat/'
path_black_hat_image = r'../image_black_hat/'
path_result_image = r'../image_result/'


def getNameImage(idx):
    image_extesion = '.jpg'
    image_name = 'image_' + addZero(idx) + str(idx) + image_extesion
    
    return image_name

def getOcr(url_image):
    
    custom_config = r'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    result_text = pytesseract.image_to_string(Image.open(url_image), config=custom_config)
    return result_text
    '''
    image = Image.open(url_image).convert('RGB')
    
    npimage = np.asarray(image).astype(np.uint8)
    
    npimage [:, :, 0] = 0
    npimage [:, :, 2] = 0
    
    im = cv2.cvtColor(npimage, cv2.COLOR_RGB2GRAY)
    
    thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    binimagem = Image.fromarray(thresh)
    nameimage = 'ocr.jpg'
    binimagem = binimagem.save(nameimage)
    
    custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(nameimage)
    
    return text'''

def saveResult(name_file, result, id):
    try:
        file = open(name_file, 'a+', encoding="utf-8")
    except FileNotFoundError:
        file = open(name_file, 'w+', encoding="utf-8")
    str_result = str(id) + ' ' + result + '\n' 
    file.writelines(str_result)
    file.close()

def addZero(idx):
    if idx < 10:
        return '000'
    elif idx < 100:
        return '00'
    return '0'

def loadImageGrayScale(idx):
    path = r'../image_plate/'
    image_name = getNameImage(idx)
    
    image_url = path + image_name
    
    image_origin = cv2.imread(image_url)
    
    image_gray = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    
    return image_gray

def whiteHatOperation(image_gray):
    image_result = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, kernel, iterations=iteration) 
    return image_result

def blackHatOperation(image_gray):
    image_result = cv2.morphologyEx(image_gray,  
                                cv2.MORPH_BLACKHAT, 
                                kernel,iterations=iteration) 
    return image_result

def gradientOperation(image_gray):
    image_white_hat = whiteHatOperation(image_gray)
    image_black_hat = blackHatOperation(image_gray)
    
    #image_result = image_gray + image_white_hat - image_black_hat
    image_op1 = cv2.add(image_gray, image_white_hat)
    image_result = cv2.subtract(image_op1, image_black_hat)
    
    return image_result

def saveImageGray(image_gray, idx): 
    image_name = getNameImage(idx)
    image_url = path_gray_image + image_name
    
    cv2.imwrite(image_url, image_gray)
    
def saveImageWhiteHat(image_gray, idx):
    image_name = getNameImage(idx)
    image_url = path_white_hat_image + image_name
    
    image_white_hat = whiteHatOperation(image_gray)
    
    cv2.imwrite(image_url, image_white_hat)

def saveImageBlackHat(image_gray, idx):
    image_name = getNameImage(idx)
    image_url = path_black_hat_image + image_name
    
    image_black_hat = blackHatOperation(image_gray)
    
    cv2.imwrite(image_url, image_black_hat)

def saveImageResult(image_gray, idx):
    image_name = getNameImage(idx)
    image_url = path_result_image + image_name
    
    image_result = gradientOperation(image_gray)
      
    cv2.imwrite(image_url, image_result)

def saveValueBoard(path, idx):
    image_name = getNameImage(idx)
    image_url = path + image_name
    
    result_text = getOcr(image_url)
    
    result_path = path + 'result.txt'
    saveResult(result_path, result_text, idx)

for idx in range(1,amount_images,1):
    print(idx)
    image_gray = loadImageGrayScale(idx)
    
    saveImageResult(image_gray,idx)       
        
    saveValueBoard(path_result_image, idx)
    