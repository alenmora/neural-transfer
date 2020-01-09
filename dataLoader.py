import torch as torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob
import os

class imageDataSet(Dataset):
    """
    data set containing a list of images
    """
    def __init__(self, root):
        self.root = root
        self.paths = []
        if os.path.isfile(self.root):
            self.paths.append(self.root)
        elif os.path.isdir(self.root):
            self.paths = glob(os.path.join(self.root, '*.jpg'))
            self.paths = self.paths + glob(os.path.join(self.root, '*.jpeg'))
            self.paths = self.paths + glob(os.path.join(self.root, '*.png'))
        else:
            print(f'Error! The root {self.root} is neither a directory nor an image file!')
            raise ValueError

    def __len__(self):
        return len(self.paths)

    def __getitem__ (self, idx):
        idx = idx % self.__len__()
        return self.paths[idx]

    def __call__(self):
        return self.paths

class dataLoader():
    """
    data loader class. Contains the datasets for style and contents. 
    If called, returns all the content images and one style image
    """
    def __init__(self,config):
        self.styleDataSet = imageDataSet(config.styles)
        self.contentDataSet = imageDataSet(config.contents)
        self.imsize = config.imageSize
        self.stylesIdx = 0
        self.contentsIdx = 0
        self.finished = False
        trans = [transforms.Resize((self.imsize,self.imsize)), transforms.ToTensor()]
        if 'vgg' in config.network:
            trans.append(transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1,1,1]))
        trans.append(lambda x: x.mul_(255))
        self.transform = transforms.Compose(trans)

    def __call__(self):
        if not self.finished:
            stylePath = self.styleDataSet[self.stylesIdx]
            contPath = self.contentDataSet[self.contentsIdx]

            style = self.getTensorFrom_(stylePath)
            cont = self.getTensorFrom_(contPath)

            self.contentsIdx += 1
            if (self.contentsIdx % len(self.contentDataSet) == 0):
                self.stylesIdx += 1

            if (self.stylesIdx % len(self.styleDataSet) == 0):
                self.finished = True

            return cont, style, contPath, stylePath
        
        else:
            return None

    def getTensorFrom_(self, path):
        im = Image.open(path)
        im = self.transform(im)
        im = im.view(1,*im.shape)
        return im

    def numOfStyles(self):
        return len(self.styleDataSet)

    def numOfContents(self):
        return len(self.contentDataSet)

    def getContents():
        cont, _ = self.__call__()
        names = self.contentDataSet()

        return cont, names

    def getStyles():
        _, styles = self.__call__()
        names = self.styleDataSet()

        return styles, names