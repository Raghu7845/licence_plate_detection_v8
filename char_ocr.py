
# from IPython.display import Image
from matplotlib import pyplot as plt

import cv2
import argparse
import sys
import numpy as np
import pandas as pd
import os.path
import time
import re

# from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
# from keras.models import Model, Sequential

import string
import easyocr
import pytesseract
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en', rec_algorithm='CRNN')

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
					'Q': '0',
					'I': '1',
					'J': '3',
					'A': '4',
					'G': '6',
					'S': '5',
					'Z': '7'}

dict_int_to_char = {'0': 'O',
					'1': 'I',
					'3': 'J',
					'4': 'A',
					'6': 'G',
					'5': 'S', 
					'7': 'Z'}



def format_license(licence):
	#TNO2BL8713
	print("inside format licence")
	print(licence)
	licence = licence.upper().replace(' ', '')
	licence = re.sub('[^a-zA-Z0-9 \n\.]', '', licence)
	licence_pattern = r"([A-Z0-9]{2})([A-Z0-9]{1,2})((?:[A-Z0-9])?(?:[A-Z0-9]*)?)([A-Z0-9]{4})"
	match = re.match(licence_pattern, licence, re.I)
	state, district_num, district_alpha, l_no = '','','',''
	result = ''
	if match:
	    state, district_num, district_alpha, l_no = match.groups()
	    print(state, district_num, district_alpha, l_no)
	else:
		print("Not the correct regex: {licence}")
		return licence
	for i in range(len(state)):
		result+= dict_int_to_char.get(state[i], state[i])

	for i in range(len(district_num)):
		result+= dict_char_to_int.get(district_num[i], district_num[i])

	for i in range(len(district_alpha)):
		result+= dict_int_to_char.get(district_alpha[i], district_alpha[i])

	for i in range(len(l_no)):
		result+= dict_char_to_int.get(l_no[i], l_no[i])

	# result = state+district_num+district_alpha+l_no
	return result


# class OCRModel():
	
# 	_loaded_model = None

# 	def __init__(self):
# 		raise RuntimeError('Call instance() instead')

# 	@classmethod
# 	def get_model(cls):
# 		if cls._loaded_model is None:
# 			cls._loaded_model = Sequential()
# 			cls._loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
# 			cls._loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
# 			cls._loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
# 			cls._loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
# 			cls._loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
# 			cls._loaded_model.add(Dropout(0.4))
# 			cls._loaded_model.add(Flatten())
# 			cls._loaded_model.add(Dense(128, activation='relu'))
# 			cls._loaded_model.add(Dense(36, activation='softmax'))

# 			# Restore the weights
# 			cls._loaded_model.load_weights('checkpoints/my_checkpoint')

# 		return cls._loaded_model



# Match contours to license plate or character template
def find_contours(dimensions, img) :

	# Find all contours in the image
	cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Retrieve potential dimensions
	lower_width = dimensions[0]
	upper_width = dimensions[1]
	lower_height = dimensions[2]
	upper_height = dimensions[3]
	
	# Check largest 5 or  15 contours for license plate or character respectively
	cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
	
	ii = cv2.imread('contour.jpg')
	
	x_cntr_list = []
	target_contours = []
	img_res = []
	for cntr in cntrs :
		# detects contour in binary image and returns the coordinates of rectangle enclosing it
		intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
		
		# checking the dimensions of the contour to filter out the characters by contour's size
		if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
			x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

			char_copy = np.zeros((44,24))
			# extracting each character using the enclosing rectangle's coordinates.
			char = img[intY:intY+intHeight, intX:intX+intWidth]
			char = cv2.resize(char, (20, 40))
			
			# Make result formatted for classification: invert colors
			char = cv2.subtract(255, char)

			# Resize the image to 24x44 with black border
			char_copy[2:42, 2:22] = char
			char_copy[0:2, :] = 0
			char_copy[:, 0:2] = 0
			char_copy[42:44, :] = 0
			char_copy[:, 22:24] = 0

			img_res.append(char_copy) # List that stores the character's binary image (unsorted)
			
	# Return characters on ascending order with respect to the x-coordinate (most-left character first)
	indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
	img_res_copy = []
	for idx in indices:
		img_res_copy.append(img_res[idx])# stores character images according to their index
	img_res = np.array(img_res_copy)
	return img_res

# Find characters in the resulting images
def segment_characters(image) :
	# Preprocess cropped license plate image
	img_lp = cv2.resize(image, (333, 75))
	# img_lp = cv2.resize(img_lp, None, fx = 2, fy = 2,  interpolation = cv2.INTER_CUBIC)
	img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
	# img_gray_lp = cv2.GaussianBlur(img_gray_lp, (5, 5), 0) 
	# img_binary_lp = img_gray_lp
	# _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	_, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_OTSU)
	img_binary_lp = cv2.erode(img_binary_lp, (3,3))
	img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

	LP_WIDTH = img_binary_lp.shape[0]
	LP_HEIGHT = img_binary_lp.shape[1]

	# Make borders white
	img_binary_lp[0:3,:] = 255
	img_binary_lp[:,0:3] = 255
	img_binary_lp[72:75,:] = 255
	img_binary_lp[:,330:333] = 255

	# Estimations of character contours sizes of cropped license plates
	dimensions = [LP_WIDTH/6,
					   LP_WIDTH/2,
					   LP_HEIGHT/10,
					   2*LP_HEIGHT/3]

	cv2.imwrite('contour.jpg',img_binary_lp)

	# Get contours within cropped license plate
	# char_list = find_contours(dimensions, img_binary_lp)

	# return char_list

def fix_dimension(img): 
	new_img = np.zeros((28,28,3))
	for i in range(3):
		new_img[:,:,i] = img
		return new_img
  
# def show_results(char):
# 	dic = {}
# 	characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# 	for i,c in enumerate(characters):
# 		dic[i] = c

# 	output = []
# 	print("inside show result")
# 	for i,ch in enumerate(char): #iterating over the characters
# 		print("inside char segmentations")
# 		img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
# 		img = fix_dimension(img_)
# 		img = img.reshape(1,28,28,3) #preparing image for the model
# 		# plt.imshow(img)
# 		# plt.title("Extracted license plate")
# 		# plt.show()
# 		cv2.imshow('Frame', img)
# 		time.sleep(60)
# 		try:
# 			y_=OCRModel.get_model().predict(img) 
# 			y_=np.argmax(y_,axis=1)
# 		except Exception as err:
# 			print(f"error while predicting CHAR: {str(err)}")
# 		character = dic[y_]
# 		output.append(character) #storing the result in a list
		
# 	plate_number = ''.join(output)
# 	raise RuntimeError("Got the results")
# 	return plate_number

def read_tesseract():
	license_plate_crop = cv2.imread('contour.jpg')
	string_list=[]
	score = 0

	# print("in tesseract")
	# string = pytesseract.image_to_string(license_plate_crop, lang ='eng')
	# string_list.append(format_license(string))

	string_list, conf = ocr.ocr(license_plate_crop, cls=False, det=False)[0][0]
	string_list = format_license(string_list)

	return string_list, conf

	# detections = reader.readtext(license_plate_crop)
	# for detection in detections:
	# 	bbox, text, score = detection

	# 	string = text.upper().replace(' ', '')
	# 	string_list.append(format_license(string))
	return "_".join(string_list), score

def read_license_plate(license_plate_crop_thresh):
	# char_segment = segment_characters(license_plate_crop_thresh)
	segment_characters(license_plate_crop_thresh)
	# plate_number = show_results(char_segment)
	plate_number, score = read_tesseract()
	return plate_number, None
