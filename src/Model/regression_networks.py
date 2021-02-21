import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import constants
from constants import *    

# Neural Network which attempts to predict total fantasy pointers
# per player for the next season. Meant to be compared to the Bagging Model
class SinglePredictorModel(nn.Module):
    def __init__(self, lr=0.0005, criterion=None):
        super(SinglePredictorModel, self).__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(46, 22)
        self.hidden1 = nn.Linear(22, 11)
        self.output = nn.Linear(11, 1)
        if criterion:
            self.criterion  = criterion
        else:
            self.criterion = nn.L1Loss()  
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        # standard norm better
        
    def forward(self, x):
        # Pass the input tensor through each layer of the network
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.output(x))

        return x
    
# Neural Network to make predictions based on stats
# calculated on a per game basis
class OverallModel(nn.Module):
    def __init__(self, lr=0.005, criterion=None):
        super(OverallModel, self).__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(44, 22)
        self.output = nn.Linear(22, 1)
        if criterion:
            self.criterion  = criterion
        else:
            self.criterion = nn.L1Loss()  
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        # standard norm better
        
    def forward(self, x):
        # Pass the input tensor through each layer of the network
        x = F.relu(self.hidden(x))
        x = F.relu(self.output(x))

        return x
        
# Neural Network to predict number of games played by
# players in the following seasons
class GamesPlayedModel(nn.Module):
    def __init__(self, lr=0.0005, criterion=None):
        super(GamesPlayedModel, self).__init__()
        self.hidden = nn.Linear(44, 22)
        self.hidden1 = nn.Linear(22, 11)
        self.output = nn.Linear(11, 1)
        if criterion:
            self.criterion  = criterion
        else:
            self.criterion = nn.SmoothL1Loss()  
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        # pass the input tensor through each layer of the network
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.output(x))
        
        return x

# Neural Network to predict passing points per game for the next season
class QBModel(nn.Module):
    def __init__(self, lr=0.01, criterion=None):
        super(QBModel, self).__init__()
        self.hidden = nn.Linear(44, 22)
        self.output = nn.Linear(22, 1)
        if criterion:
            self.criterion  = criterion
        else:
            self.criterion = nn.SmoothL1Loss()  
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        #minmax norm works best

    def forward(self, x):
        # pass the input tensor through each layer of the network
        x = F.relu(self.hidden(x))
        x = F.relu(self.output(x))
        
        return x

# Neural Network to predict rushing points per game for the next season
class RBModel(nn.Module):
    def __init__(self, lr=.001, criterion=None):
        super(RBModel, self).__init__()
        self.hidden = nn.Linear(44, 22)
        self.hidden1 = nn.Linear(22, 11)
        self.output = nn.Linear(11, 1)
        if criterion:
            self.criterion  = criterion
        else:
            self.criterion = nn.SmoothL1Loss()    
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        #standard norm works best

    def forward(self, x):
        # pass the input tensor through each layer of the network
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.output(x))
        
        return x

# Neural Network to predict receiving points per game for the next season
class WRModel(nn.Module):
    def __init__(self, lr=.0005, criterion=None):
        super(WRModel, self).__init__()
        self.hidden = nn.Linear(44, 22)
        self.output = nn.Linear(22, 1)
        if criterion:
            self.criterion  = criterion
        else:
            self.criterion = nn.L1Loss()  
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        # pass the input tensor through each layer of the network
        x = F.relu(self.hidden(x))
        x = F.relu(self.output(x))
        
        return x



