import argparse
import os

parser = argparse.ArgumentParser('style-transfer')

############################
#  Paths
############################

parser.add_argument('--styles', type=str, default='./styles/')               # Image (or folder of images) that determines the drawing style(s) of the output
parser.add_argument('--contents', type=str, default='./contents/')           # Image (or folder of images) that determines the content(s) of the output
parser.add_argument('--outputFolder', type=str, default='./generated/')          # Folder where to output the generated images. It is created if it doesn't exist already

############################
#  Generator options
############################

parser.add_argument('--nLoops', type=int, default=500)                                                             # Number of training loops for the images
parser.add_argument('--network', type=str, default='vgg19')                                                        # Which object detection network to use. Default is VGG19. Possible options are AlexNet, VGG16 and VGG19
parser.add_argument('--contentLayers', type=str, default='relu_1')                                                 # Layer to use to calculate the content loss 
parser.add_argument('--styleLayers', type=str, default='relu_2, relu_3, relu_4, relu_6, relu_8')                   # Initial and final layers to use for the calculation of the style loss

parser.add_argument('--alpha', type=float, default=1.)                           # Weight of the content loss term
parser.add_argument('--beta', type=float, default=1000.)                         # Weight of the style loss term
parser.add_argument('--poolingStyle', type=str, default='Average')               # Which algorithm to use for max pooling. Possible options are Average or Max
parser.add_argument('--weightStyle', type=str, default='Geometric')              # How to calculate the weight for each layer loss in the style loss. Geometric assigns a weight of 1 for the first layer, and keeps dividing by two the weight (normalized at the end). The other option is Uniform, where all layers weight the same
parser.add_argument('--startFromWhiteNoise', action ='store_true')                # Wether to start from white noise or from the contents image

############################
#  Output
############################

parser.add_argument('--logLoss', action='store_true')                              # Wether to log loss information or not
parser.add_argument('--saveSnapshotEvery', type=int, default=0)                  # Wether to take snapshots during the training or not
parser.add_argument('--imageSize', type=int, default=512)                        # Size of the output images 
parser.add_argument('--showImagesPer', type=str, default='Style')                # Plot a grid with all the images generated. The argument correponds to the rows of the grid. Style plots all contents for the same style in one row. Content plots all style for the same content in one row.

config, _ = parser.parse_known_args()