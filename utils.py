import torch
import numpy as np 
import math
import os
import torchvision.transforms as transforms


def makeImagesGrid(tensor, nrow=8, padding=2, pad_value=0):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor

    if isinstance(tensor, list):
        tensor = torch.cat(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    
    return grid.detach()
    
def saveImage(tensor, filename, nrow=8, padding=2, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    tensor = tensor.cpu()
    grid = makeImagesGrid(tensor, nrow=nrow, padding=padding, pad_value=pad_value)
    toIm = transforms.ToPILImage() 
    im = toIm(grid)
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

def keepOriginalColors(content, generated):
    contentChannels = list(content.convert('YCbCr').split())
    generatedChannels = list(generated.convert('YCbCr').split())
    contentChannels[0] = generatedChannels[0]
    return Image.merge('YCbCr', contentChannels).convert('RGB')