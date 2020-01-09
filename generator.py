import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np 
import PIL
from PIL import Image
import torchvision.transforms as transforms
from dataLoader import dataLoader
import utils
import lossModules
from config import config
import os

class generator:
    """
    generator class
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.nLoops = int(config.nLoops)

        self.modelsDict = {
                            'AlexNet':  models.alexnet,
                            'vgg16':    models.vgg16,
                            'vgg19':    models.vgg19,
                          }

        self.nn = self.modelsDict[config.network]
        
        self.nn = self.nn(pretrained=True).features.to(device = self.device)

        self.logLoss = config.logLoss

        self.saveSnapshotEvery = config.saveSnapshotEvery

        self.dataLoader = dataLoader(config)

        self.useWhiteNoise = config.startFromWhiteNoise

        self.contentLayers = "".join(config.contentLayers.split(" ")).split(',')
        self.styleLayers = "".join(config.styleLayers.split(" ")).split(',')

        self.alpha = config.alpha
        self.beta = config.beta

        self.poolingDict = {
                            'Max': nn.MaxPool2d,        
                            'Average': nn.AvgPool2d
                           }

        self.contentsLossModules = [] 
        self.stylesLossModules = []
        self.showImages = self.showImagesPerContent if config.showImagesPer == 'Content' else self.showImagesPerStyle

        self.pooling = self.poolingDict[config.poolingStyle]

        self.gram = lossModules.GramMatrix().to(device = self.device)

        self.genIm = {}
        self.styles = [s.split('/')[-1].split('.')[0] for s in self.dataLoader.styleDataSet()]
        self.contents = [c.split('/')[-1].split('.')[0] for c in self.dataLoader.contentDataSet()]

        for style in self.styles:
            self.genIm[style] = {}

        self.outputFolder = utils.createDir(config.outputFolder)

        trans = [transforms.Lambda(lambda x: x.mul_(1/255.))]

        if 'vgg' in config.network:
            trans.append(transforms.Normalize(mean=[-0.48501961, -0.45795686, -0.40760392], std=[1,1,1]))

        trans.append(transforms.Lambda(lambda x: x.clamp_(0,1)))

        self.trans = transforms.Compose(trans)

    def checkName(self, name, model, content, style):
        """
        Check if the given module name is in the content module list,
        and in the style module list. If so, add the corresponding loss
        module after it
        """
        i = name.split('_')[-1]

        if name in self.contentLayers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = lossModules.ContentLoss(target, self.alpha)
            model.add_module(f"content_loss_layer_{i}", content_loss)
                    
            self.contentsLossModules.append(content_loss)

        if name in self.styleLayers:
            # add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = self.gram.forward(target_feature)
            style_loss = lossModules.StyleLoss(target_feature_gram, self.beta)
            model.add_module(f"style_loss_layer_{i}", style_loss)
                
            self.stylesLossModules.append(style_loss)

    def loadNeuralNetwork(self, content, style):
        """
        Import the pretrained NN, up to the last needed layer, 
        and include the necessary loss modules
        """
        model = nn.Sequential().to(device = self.device)
        i = 1
        for layer in list(self.nn):
            if (len(self.contentsLossModules) <= len(self.contentLayers) or len(self.stylesLossModules) <= len(self.styleLayers)):
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.modules.conv.Conv2d):
                    name = "conv_" + str(i)
                    model.add_module(name, layer)
                    self.checkName(name, model, content, style)
            

                elif isinstance(layer, nn.ReLU):
                    name = "relu_" + str(i)
                    model.add_module(name, layer)
                    self.checkName(name, model, content, style)
                    i += 1

                elif isinstance(layer, nn.MaxPool2d):
                    name = "pool_" + str(i)
                    model.add_module(name, self.pooling(kernel_size=layer.kernel_size, stride=layer.stride, padding = layer.padding))

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

    def train(self, content, style, contentN, styleN):
        """
        Perform training for one content and style
        """
        print(f'Training {contentN} to look like {styleN}...')

        inputImage = content.clone()
        if self.useWhiteNoise:
            inputImage.data = torch.randn(inputImage.data.size(), device = self.device)

        inputImage.requires_grad_(True)

        optimizer = torch.optim.LBFGS([inputImage])

        self.loadNeuralNetwork(content, style)

        n = [0]

        while n[0] <= self.nLoops:
            
            def closure():
                optimizer.zero_grad()
                
                self.model.forward(inputImage)
            
                closs = 0.
                sloss = 0.

                for cl in self.contentsLossModules:
                    closs += cl.getLoss()
            
                for sl in self.stylesLossModules:
                    sloss += sl.getLoss()

                loss = closs + sloss

                loss.backward()

                n[0] += 1

                if self.logLoss and n[0] % 50 == 49:
                    print(f"Iteration {n[0]+1:5d}/{self.nLoops}; Style loss: {sloss:9.4f}; Contents loss: {closs:9.4f}; Total loss: {loss:9.4f}")

                if self.saveSnapshotEvery > 0 and n[0] % self.saveSnapshotEvery == self.saveSnapshotEvery-1:
                    filename = f'{contentN}_as_{styleN}_{n[0]+1}.png'
                    filename = os.path.join(self.outputFolder,filename)
                    output = inputImage.clone().detach().squeeze()
                    utils.saveImage(self.trans(output), filename)

                return loss

            optimizer.step(closure)

        output = inputImage.clone().squeeze()
        self.genIm[styleN][contentN] = self.trans(output)

    def trainAll(self):
        """
        Train one network per each content-style pair,
        show the results, and save them
        """
        group = self.dataLoader()

        while group:
            content, style, contentN, styleN = group
            contentN = contentN.split('/')[-1].split('.')[0]
            styleN = styleN.split('/')[-1].split('.')[0]

            self.train(content, style, contentN, styleN)

            group = self.dataLoader()
        
        #self.showImages()

        self.saveImages()

    def showImages_(self, images, nrows = None, padding = 5):
        """
        Internal function to show a group of images in a grid
        with nrows and a padding between images
        """
        if nrows == None:
            n = images.size(0)
            nrows = 1
            if math.sqrt(n) == int(math.sqrt(n)):
                nrows = math.sqrt(n)
    
            elif n > 5:
                i = int(math.sqrt(n))
                while i > 2:
                    if (n % i) == 0:
                        nrows = i
                        break

                    i = i-1
        
        grid = utils.makeImagesGrid(images, nrows, padding = padding)
    
        plt.imshow(grid.numpy().transpose(1,2,0))

        plt.show(block=True)

    def showImagesPerContent(self):
        """
        Show all the images generated, with each row
        corresponding to the same content in different styles
        """
        imageList = []
        for content in self.contents:
            stylesList = []
            for style in self.styles:
                stylesList.append(self.genIm[style][content]) 

            imageList.append(torch.stack(stylesList, dim=0))
        
        self.showImages_(imageList, nrows = len(self.contents))

    def showImagesPerStyle(self):
        """
        Show all the images generated, with each row
        corresponding to the style applied to different contents
        """
        imageList = []
        for style in self.styles:
            contentsList = []
            for content in self.contents:
                contentsList.append(self.genIm[style][content]) 

            imageList.append(torch.stack(contentsList, dim=0))

        self.showImages_(imageList, nrows = len(self.styles))

    def saveImages(self):
        """
        Save the generated images in the outputFolder,
        """
        for style, dic in self.genIm.items():
            for content, im in dic.items():
                filename = f'{content}_as_{style}.jpg'
                filename = os.path.join(self.outputFolder,filename)
                utils.saveImage(im, filename) 

if __name__ == "__main__":
    gen = generator(config)
    print('Generator instantiated. Proceeding to train...')
    gen.trainAll()
