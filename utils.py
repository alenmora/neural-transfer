import torch
import numpy as np 
import math
import os
import torchvision.transforms as transforms
from PIL import Image
    
def saveImage(tensor, filename, original = None, keepColors = False, nrow=8, padding=2, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    if not original: keepColors = False
    tensor = tensor.cpu()
    size = (tensor.size(1), tensor.size(2))
    toIm = transforms.ToPILImage()
    im = toIm(tensor)
    if keepColors:
        im = list(im.convert('YCbCr').split())
        orIm = list(Image.open(original).resize(size, resample=Image.LANCZOS).convert('YCbCr').split())
        orIm[0] = im[0]
        im = Image.merge('YCbCr', orIm).convert('RGB')
    
    im.save(filename)

def createDir(dir):
    """
    Create directory
    """
    try: 
        os.makedirs(dir)
        print(f'Created new folder at {dir}')
    except FileExistsError: 
        print(f'Using previously created folder {dir}')
    return dir

def saveGif(dir):
    """
    Takes a group of images and saves them in a gif
    """
    paths = glob(os.path.join(dir, '*.jpg'))
    paths = paths + glob(os.path.join(dir, '*.jpeg'))
    paths = paths + glob(os.path.join(dir, '*.png'))

    ims = []
    name = '.gif'
    for im in paths:
        ims.append(Image.open(im))
        name = ims[0].split('/')[0]+'/'+ims[0].split('/')[-1].split('.')[0]+ name
    ims[0].save(name, save_all=True, append_images=ims[1:], duration=500,loop=1)