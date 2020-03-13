import torch
from torch import nn

class ContentLoss(nn.Module):
    """
    Transparent (output = input) module to be inserted after the 
    content layer(s) in order to calculate the content loss
    """
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input * self.weight, self.target)
        return input

    def getLoss(self):
        return self.loss

class GramMatrix(nn.Module):
    """
    Module that calculates the Gram matrix of the input 
    (Gram matrix = A x A^T), where A is a matrix with each row
    corresponding to one unrolled channel of the input.
    It corresponds to the scalar product of all pairs of channels
    """
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size (1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)

class StyleLoss(nn.Module):
    """
    Transparent (output = input) module inserted to be inserted after the 
    style layer(s) in order to calculate the style loss
    """
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.G = self.gram.forward(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion.forward(self.G, self.target)
        return input

    def getLoss(self):
        return self.loss