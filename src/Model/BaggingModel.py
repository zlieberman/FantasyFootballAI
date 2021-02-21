import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import constants
from constants import *                                         

# Neural Network using input from specific feature
# models to make an overall prediction on a players
# performance
class BaggingModel(nn.Module):
    def __init__(self, criterion, lr, gpmodel, qbmodel, rbmodel, wrmodel, ovmodel, spmodel):
        super(BaggingModel, self).__init__()

        # the ensemble of networks to bag
        self.qbmodel = qbmodel
        self.rbmodel = rbmodel
        self.wrmodel = wrmodel
        self.gpmodel = gpmodel
        self.ovmodel= ovmodel
        self.spmodel = spmodel

        # where to split the data
        self.gp_idx = len(GPNAMES)
        self.qb_idx = self.gp_idx + len(QBNAMES)
        self.rb_idx = self.qb_idx + len(RBNAMES)
        self.wr_idx = self.rb_idx + len(WRNAMES)
        self.ov_idx = self.wr_idx + len(OVNAMES)
        self.sp_idx = self.ov_idx + len(OVNAMES)

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(6, 3)
        self.output = nn.Linear(6, 1)
        
        self.criterion  = criterion
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        
    def forward(self, x):
        """
        # separate the input into appropriate inputs for each model in the ensemble
        # Note: torch.stack() concatenates tensors in new dimension (create a list of lists)
        #            torch.cat() concatenates tensors in existing dimension (continue a single list)
        gp = torch.stack(list((torch.Tensor(t[0:self.gp_idx]) ) for t in x))
        qb = torch.stack(list((torch.Tensor(t[self.gp_idx:self.qb_idx]) ) for t in x))
        rb = torch.stack(list((torch.Tensor(t[self.qb_idx:self.rb_idx]) ) for t in x))
        wr = torch.stack(list((torch.Tensor(t[self.rb_idx:self.wr_idx]) ) for t in x))
        ov = torch.stack(list((torch.Tensor(t[self.wr_idx:self.ov_idx]) ) for t in x))
        sp = torch.stack(list((torch.Tensor(t[self.ov_idx:self.sp_idx]) ) for t in x))
        """
        # fill output tensor for each of the constituent models
        gp_out = self.gpmodel(x)
        qb_out = self.qbmodel(x)
        rb_out = self.rbmodel(x)
        wr_out = self.wrmodel(x)
        ov_out = self.ovmodel(x)
        sp_out = self.spmodel(x)
        # use ensemble results to create a tensor of each model's output per input player
        y = torch.stack(list((torch.cat([gp_out[i],qb_out[i],rb_out[i],wr_out[i],ov_out[i],sp_out[i]])) for i in range(len(x))))
        
        # Pass the input tensor through each of our operations
        #y = F.relu(self.hidden(y))
        y = F.relu(self.output(y))
        
        return y
