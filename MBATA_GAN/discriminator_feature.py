import torch.nn as nn
import torch.nn.functional as F
import Config1 as config
from modulesD import *

class FCDiscriminator_feature(nn.Module):

	def __init__(self):
		super(FCDiscriminator_feature, self).__init__()
		vis = True
		config_vit = config.get_CTranS_config()

		self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
		self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
		self.conv3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
		self.classifier = nn.Conv2d(512, 1, kernel_size=1, stride=1)

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
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x
