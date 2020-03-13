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

        self.pooling = self.poolingDict[config.poolingStyle]

        self.gram = lossModules.GramMatrix().to(device = self.device)

        self.genIm = {}

        for style in self.dataLoader.styleDataSet():
            self.genIm[style] = {}

        self.outputFolder = utils.createDir(config.outputFolder)

        trans = [transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])]
        
        trans.append(transforms.Lambda(lambda x: x.clamp_(0,1)))

        self.trans = transforms.Compose(trans)

        self.keepColors = config.keepColors

    def checkName(self, name, model, content, style):
        """
        Check if the given module name is in the content module list,
        and in the style module list. If so, add the corresponding loss
        module after it
        """
        i = name.split('_')[-1]

        if name in self.contentLayers:
            # add content loss:
            target = model.forward(content)
            content_loss = lossModules.ContentLoss(target, self.alpha)
            model.add_module(f"content_loss_layer_{i}", content_loss)
                    
            self.contentsLossModules.append(content_loss)

        if name in self.styleLayers:
            # add style loss:
            target_feature = model.forward(style)
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

        for param in model.parameters():
            param.requires_grad = False

        return model

    def train(self, content, style, contentN, styleN):
        """
        Perform training for one content and style
        """

        contentNs = self.getImageName(contentN)
        styleNs = self.getImageName(styleN)

        print(f'Training {contentNs} to look like {styleNs}...')

        inputImage = content.clone().to(self.device)
        if self.useWhiteNoise:
            inputImage = torch.randn(inputImage.data.size(), device = self.device)

        inputImage.requires_grad_(True)

        optimizer = torch.optim.LBFGS([inputImage])

        model = self.loadNeuralNetwork(content, style)

        n = [0]

        while n[0] <= self.nLoops:
            
            def closure():
                optimizer.zero_grad()
                
                model.forward(inputImage)
            
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
                    filename = f'{contentNs}_as_{styleNs}_{n[0]+1}.png'
                    filename = os.path.join(self.outputFolder,filename)
                    output = inputImage.clone().detach().squeeze()
                    utils.saveImage(self.trans(output), filename, original = contentN, keepColors = self.keepColors)

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
        
            self.train(content, style, contentN, styleN)

            group = self.dataLoader()

        self.saveImages()

    def saveImages(self):
        """
        Save the generated images in the outputFolder,
        """
        for styleN, dic in self.genIm.items():
            for contentN, im in dic.items():
                filename = f'{self.getImageName(contentN)}_as_{self.getImageName(styleN)}.jpg'
                filename = os.path.join(self.outputFolder,filename)
                utils.saveImage(im, filename, original = content, keepColors = self.keepColors) 

    def getImageName(self, imagePath):
        return imagePath.split('/')[-1].split('.')[0]
    
if __name__ == "__main__":
    gen = generator(config)
    print('Generator instantiated. Proceeding to train...')
    gen.trainAll()
