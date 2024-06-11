import os
import sys
from pathlib import Path

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import cv2
import glob
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(1143)


def populate_train_list(lowlight_images_path):
	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	train_list = image_list_lowlight
	random.shuffle(train_list)
	return train_list


def populate_val_list(highlight_images_path):
	image_list_highlight = glob.glob(highlight_images_path + "*.jpg")
	val_list = image_list_highlight
	random.shuffle(val_list)
	return val_list

def get_filenames(image_folder_path, img_extensions = ['.jpg', '.jpeg']):
	folder_path = Path(f"{image_folder_path}")

	# 이미지 파일 확장자 리스트를 만듭니다.
	image_extensions = img_extensions
	# 폴더 내 이미지 파일명을 가져옵니다.
	image_files = [file.name for file in folder_path.iterdir() if file.suffix.lower() in image_extensions]
	return image_files

def get_file_path(folder_path, img_name):
	assert isinstance(folder_path, str), f"Folder path is not string, {type(folder_path)}"
	assert isinstance(img_name, str), f"Image name is not string, {type(img_name)}"
	full_path = Path(folder_path) / img_name
	assert full_path.exists(), f"There is no file with name {str(full_path)}"
	return full_path

# class lowlight_loader(data.Dataset):
# 	def __init__(self, lowlight_images_path):
# 		self.train_list = populate_train_list(lowlight_images_path)
# 		self.size = 512

# 		self.data_list = self.train_list
# 		print("Total training examples:", len(self.train_list))


# 	def __getitem__(self, index):

# 		data_lowlight_path = self.data_list[index]
		
# 		data_lowlight = Image.open(data_lowlight_path)
		
# 		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
# 		data_lowlight = (np.asarray(data_lowlight)/255.0) 
# 		data_lowlight = torch.from_numpy(data_lowlight).float()

# 		return data_lowlight.permute(2,0,1)

# 	def __len__(self):
# 		return len(self.data_list)

class low_high_light_loader(data.Dataset):
	def __init__(self, lowlight_images_path, hightlight_images_path):
		self.lowlight_images_path = lowlight_images_path
		self.highlight_images_path = hightlight_images_path
		self.img_list = get_filenames(lowlight_images_path)
		self.high_img_list = get_filenames(hightlight_images_path)
		assert self.img_list == self.high_img_list, f"lowlight images are not match highlight images"
		self.size = 512

		self.data_list = self.img_list
		print("Total training examples:", len(self.img_list))


	def __getitem__(self, index):

		img_name = self.data_list[index]
		
		lowlight_image_path = get_file_path(self.lowlight_images_path, img_name)
		highlight_image_path = get_file_path(self.highlight_images_path, img_name)
		data_lowlight = Image.open(str(lowlight_image_path))
		data_highlight = Image.open(str(highlight_image_path))
		
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()
		data_lowlight = data_lowlight.permute(2,0,1)

		data_highlight = data_highlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_highlight = (np.asarray(data_highlight)/255.0) 
		data_highlight = torch.from_numpy(data_highlight).float()
		data_highlight = data_highlight.permute(2,0,1)
		return data_lowlight, data_highlight

	def __len__(self):
		return len(self.data_list)
