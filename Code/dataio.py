import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from model import *

class ScalarDataSet():
	def __init__(self,args):
		self.device = torch.device("cuda:0" if args.cuda else "cpu")
		self.dataset = args.dataset
		self.crop = args.crop
		self.croptimes = args.croptimes
		if self.dataset == 'Combustion':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.cropsize = [64,192,32]
			self.samples = [i for i in range(1,41)]
			self.s = '../Data/Combustion/MF-'   
			self.t = '../Data/Combustion/Chi-'                             

	def ReadData(self):
		self.source = []
		self.target = []
		for i in self.samples:
			print(i)
			s = np.zeros((1,self.dim[0],self.dim[1],self.dim[2]))
			t = np.zeros((1,self.dim[0],self.dim[1],self.dim[2]))
			d = np.fromfile(self.s+'{:04d}'.format(i)+'.iw',dtype='<f')

			d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
			d = d.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
			s[0] = d
			self.source.append(s)
			
			o = np.fromfile(self.t+'{:04d}'.format(i)+'.iw',dtype='<f')
			o = 2*(o-np.min(o))/(np.max(o)-np.min(o))-1
			o = o.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
			t[0] = o
			self.target.append(t)

		self.source = np.asarray(self.source)
		self.target = np.asarray(self.target)

	def TrainingData(self):
		if self.crop == 'yes':
			a = []
			o = []
			for k in range(1,len(self.samples)+1):
				n = 0
				while n < self.croptimes:
					if self.dim[0] == self.cropsize[0]:
						x = 0
					else:
						x = np.random.randint(0,self.dim[0]-self.cropsize[0])

					if self.dim[1] == self.cropsize[1]:
						y = 0
					else:
						y = np.random.randint(0,self.dim[1]-self.cropsize[1])

					if self.dim[2] == self.cropsize[2]:
						z = 0
					else:
						z = np.random.randint(0,self.dim[2]-self.cropsize[2])

					c0 = self.source[k-1][:,x:x+self.cropsize[0],y:y+self.cropsize[1],z:z+self.cropsize[2]]
					o.append(c0)

					c1 = self.target[k-1][:,x:x+self.cropsize[0],y:y+self.cropsize[1],z:z+self.cropsize[2]]
					a.append(c1)

					n+= 1

			o = np.asarray(o)
			a = np.asarray(a)
			a = torch.FloatTensor(a)
			o = torch.FloatTensor(o)
		else:
			a = torch.FloatTensor(self.target)
			o = torch.FloatTensor(self.source)
		dataset = torch.utils.data.TensorDataset(o,a)
		train_loader = DataLoader(dataset=dataset,batch_size=1, shuffle=True)
		return train_loader







