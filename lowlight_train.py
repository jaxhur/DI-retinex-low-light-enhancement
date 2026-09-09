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
from torch.distributions.normal import Normal
import sys 
import random
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config, exp, Gaussian=None):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	net = mymodel.enhance_net_nopool().cuda()

	net.apply(weights_init)
	if config.load_pretrain == True:
	    net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)

	optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	net.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):
			img_lowlight,img_gauss, imgfile = img_lowlight
			img_gauss = img_gauss.cuda()
			img_lowlight = img_lowlight.cuda()
			enhanced_image, c, b, k  = net(img_lowlight)

		
			
			mask = enhanced_image<1
			imgname = imgfile[0].split('/')[-1] 
			
			Gaussian = Normal(torch.tensor([exp]), torch.tensor([0.001]))
			target_exposure = Gaussian.sample().cuda() # 0.4
			ratio = target_exposure / img_lowlight.detach().mean()

			# print(ratio.reshape(-1))
			loss1 = (mask * (img_gauss**2.2 - (1/ratio)**2.2 * (enhanced_image)**2.2)).abs().mean()
			loss2 = (k * b - k + b + 1).var((2,3)).mean() # (k * b - k + b + 1).var().mean() 
			loss = loss1 + loss2
			# print((k * b - k + b + 1).mean((2,3)).mean().item())
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at epoch %04d iteration" % epoch, iteration+1, ":", loss.item(), enhanced_image.mean().item(), imgname,)
				torchvision.utils.save_image(enhanced_image, 'tmp.png')
		if ((epoch+1) % config.snapshot_epoch) == 0:
			torchvision.utils.save_image(enhanced_image, 'tmp.png')
			torch.save(net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 	
			torch.save(net.state_dict(), config.snapshots_folder + "latest" + '.pth') 		



def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
	# for i in range(0,100):
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="/home/ssq/Desktop/phd/data/llie/LOL-v2/Real_captured/Test/Low/") # '/home/ssq/Desktop/phd/data/llie/LOLdataset/our485/low')#
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_epoch', type=int, default=100)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
	parser.add_argument('--seed', type=int, default=1149)
	parser.add_argument('--exposure', type=float, default=0.5)

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	seed = config.seed
	exp = config.exposure
	print("seed", seed)
	print("expo", exp)
	set_seed(seed)
	train(config, exp)

			