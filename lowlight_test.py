import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import mymodel
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from natsort import natsorted

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('size and num：{:.5f}MB'.format(all_size), param_sum)
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)
 
def lowlight(image_path, net, save_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path).convert('RGB')

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	
	start = time.time()
	enhanced_image,_,_,_ = net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	image_name = image_path.split('/')[-1]
	result_path = os.path.join(save_path, image_name)
	
	
	torchvision.utils.save_image(enhanced_image, result_path)

def test(config):
	with torch.no_grad():
		filePath = config.lowlight_images_path
		save_path = config.save_path
		os.makedirs(save_path, exist_ok=True)
		net = mymodel.enhance_net_nopool(config.out_ch, config.inner_ch).cuda()
		net.load_state_dict(torch.load(config.model_path))
		getModelSize(net)
		file_list = natsorted(os.listdir(filePath))
		print(len(file_list))
		for image in file_list:
			# image = image
			print(image)
			lowlight(os.path.join(filePath, image), net, save_path)

if __name__ == '__main__':
# test_images
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="/home/ssq/Desktop/phd/data/llie/LOL-v2/Real_captured/Test/Low/")
	parser.add_argument('--model_path', type=str, default= "weights/latest (lolv2 21.35).pth")
	parser.add_argument('--save_path', type=str, default="result/lolv2")
	parser.add_argument('--out_ch', type=int, default=6)
	parser.add_argument('--inner_ch', type=int, default=64)
	config = parser.parse_args()

	test(config)

	# if dataset=="v1":
	# 	save_path = 'result/lolv1_small'
	# 	filePath = '/home/ssq/Desktop/phd/data/llie/LOLdataset/eval15/low'
	# elif dataset=="v2":
	# 	save_path = 'result/lolv2'
	# 	filePath = '/home/ssq/Desktop/phd/data/llie/LOL-v2/Real_captured/Test/Low/'
	# elif dataset=="darkface":
	# 	save_path = 'result/darkface' if epoch is None else 'result/darkface_{}'.format(epoch)
	# 	filePath = '/home/ssq/Desktop/phd/data/llie/DarkFace/DarkFace_Train_2021/image'