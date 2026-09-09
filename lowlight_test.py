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
from evaluation_metrics import (
	FullReferenceMetrics,
	build_test_pairs,
	calculate_mean_metrics,
	load_rgb_tensor,
	write_metric_csv,
)

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
 
def lowlight(image_path, net, source_root, save_path, device):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = load_rgb_tensor(image_path, device)

	
	start = time.time()
	enhanced_image,_,_,_ = net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	# 保留输入目录的相对路径，避免递归数据集内同名文件相互覆盖。
	image_name = os.path.relpath(image_path, source_root)
	result_path = os.path.join(save_path, image_name)
	os.makedirs(os.path.dirname(result_path), exist_ok=True)
	
	
	torchvision.utils.save_image(enhanced_image, result_path)
	return enhanced_image

def test(config):
	with torch.no_grad():
		filePath = config.lowlight_images_path
		save_path = config.save_path
		os.makedirs(save_path, exist_ok=True)
		device = torch.device('cuda')
		net = mymodel.enhance_net_nopool(config.out_ch, config.inner_ch).to(device)
		net.load_state_dict(torch.load(config.model_path, map_location=device))
		net.eval()
		getModelSize(net)
		image_pairs = build_test_pairs(filePath, config.gt_images_path)
		print(len(image_pairs))

		metric_calculator = None
		metric_values = []
		if config.gt_images_path:
			metric_calculator = FullReferenceMetrics(device)

		for image_name, lowlight_path, gt_path in image_pairs:
			# image = image
			print(image_name)
			enhanced_image = lowlight(lowlight_path, net, filePath, save_path, device)
			if metric_calculator is not None:
				gt_image = load_rgb_tensor(gt_path, device)
				metric_values.append(metric_calculator.calculate(enhanced_image, gt_image))

		if metric_values:
			mean_metrics = calculate_mean_metrics(metric_values)
			if config.metrics_csv:
				metrics_csv = config.metrics_csv
			elif os.path.basename(os.path.normpath(save_path)) == 'enhanced':
				metrics_csv = os.path.join(os.path.dirname(os.path.normpath(save_path)), 'metric.csv')
			else:
				metrics_csv = os.path.join(save_path, 'metric.csv')
			write_metric_csv(
				metrics_csv,
				config.experiment_name,
				config.dataset_name,
				mean_metrics,
				len(metric_values),
				config.model_path,
				save_path,
			)
			print(
				'Average metrics: PSNR={:.4f}, RGB SSIM={:.4f}, LPIPS-Alex-v0.1={:.4f}'.format(
					mean_metrics['psnr'], mean_metrics['rgb_ssim'], mean_metrics['lpips']
				)
			)
			print('Metrics saved to {}'.format(metrics_csv))

if __name__ == '__main__':
# test_images
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="/home/ssq/Desktop/phd/data/llie/LOL-v2/Real_captured/Test/Low/")
	parser.add_argument('--gt_images_path', type=str, default=None,
		help='正常曝光 GT 图像根目录；传入后按相对路径计算全参考指标。')
	parser.add_argument('--model_path', type=str, default= "weights/latest (lolv2 21.35).pth")
	parser.add_argument('--save_path', type=str, default="result/lolv2")
	parser.add_argument('--out_ch', type=int, default=6)
	parser.add_argument('--inner_ch', type=int, default=64)
	parser.add_argument('--experiment_name', type=str, default='di_retinex')
	parser.add_argument('--dataset_name', type=str, default='unknown')
	parser.add_argument('--metrics_csv', type=str, default=None,
		help='指标 CSV 路径；省略时写入 save_path/metric.csv，若 save_path 以 enhanced 结尾则写入其父目录。')
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
