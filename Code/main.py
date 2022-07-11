import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.parallel
import torch.nn.utils.spectral_norm as spectral_norm
import numbers
import math
import os
from torch import Tensor
from torch.nn import Parameter
import yaml
from dataio import *
from model import *
from train import *


parser = argparse.ArgumentParser(description='PyTorch Implementation of V2V')
parser.add_argument('--lr_G', type=float, default=1e-4, metavar='LR',
                    help='learning rate of G')
parser.add_argument('--lr_D', type=float, default=4e-4, metavar='LR',
                    help='learning rate of D')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default = 'Combustion',
                    help='dataset')
parser.add_argument('--mode', type=str, default='train' ,
                    help='training or inference')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='number of crop times per data')
parser.add_argument('--crop', type=str, default='yes', metavar='N',
                    help='whether to cop data')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}



def main():
    if args.mode == 'train':
        DataSet = ScalarDataSet(args)
        DataSet.ReadData()

        torch.manual_seed(np.random.randint(int(2**32)-1))
        np.random.seed(np.random.randint(int(2**32)-1))

        Model = V2V()
        D = Dis()

        if args.cuda:
            Model.cuda()
            D.cuda()
        Model.apply(weights_init_kaiming)

        D.apply(weights_init_kaiming)

        train(Model,D,DataSet,args)

    elif args.mode == 'inf':
        DataSet = ScalarDataSet(args)
        inf(args,DataSet)


if __name__== "__main__":
    main()
    