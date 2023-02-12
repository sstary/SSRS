import torch.nn as nn
import torch.nn.functional as F
import Config1 as config
from modulesD import *

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()
		vis = True
		config_vit = config.get_CTranS_config()
		self.mtcs1 = ChannelTransformer_single(config_vit, vis, 16, channel_num=[64, 128, 256, 512, 512],
											   patchSize=[1,1,1,1,1])

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x, _ = self.mtcs1(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x
