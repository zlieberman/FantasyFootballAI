import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import constants
from constants import *

# stack a classification model and a regression model trained on the same
# data to make a final prediction
class StackedModel(nn.Module):
    def __init__(self, classification_model, lr=0.01, criterion=nn.SmoothL1Loss()):
        super(StackedModel, self).__init__()

        self.input = nn.Linear(3, 2)
        self.hidden = nn.Linear(4, 2)
        self.output = nn.Linear(2, 1)

        self.softmax = nn.Softmax(dim=1)

        self.classification_model = classification_model

        self.criterion = criterion
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x):
        x = self.classification_model(x)
        x = F.relu(self.input(x))
        x = self.output(x)
        #x = self.relu2(x)

        return x
        
# model to predict probability of player scoring within one of
# 9 possible score intervals
class ClassificationModel(nn.Module):
    def __init__(self, lr=0.02, criterion=nn.CrossEntropyLoss()):
        super(ClassificationModel, self).__init__()

        self.input = nn.Linear(46, 23)
        self.relu1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.output = nn.Linear(23, 3)
        self.relu2 = nn.ReLU()
        

        self.softmax = nn.Softmax(dim=1)

        self.criterion = criterion
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True)

    def forward(self, x):
        
        x = self.input(x)
        x = self.dropout(x)
        x = self.relu1(x)
        
        x = F.relu(self.output(x))
        x = self.relu2(x)
        
        # apply softmax to x
        x = self.softmax(x)

        return x

# model to predict probability of player scoring within one of
# 17 possible score intervals
class BigClassificationModel(nn.Module):
    def __init__(self, lr, criterion):
        super(BigClassificationModel, self).__init__()

        self.input = nn.Linear(46, 23)
        self.output = nn.Linear(23, 6)

        self.criterion = criterion

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.output(x))

        # apply softmax to x
        x = F.softmax(x, dim=1)

        return x
