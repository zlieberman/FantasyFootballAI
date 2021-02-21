import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import constants
from constants import *      

# class to create the dataset which will be used to
# create the PyTorch DataLoader where everything
# will be stored. 
class FFDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.feature_tensor =  torch.from_numpy(np.array(features)).float()
        self.targets = targets
        self.target_tensor = torch.from_numpy(np.array(targets)).float()
        # standardization fields
        self.feature_mean = 0
        self.feature_stddev = 0
        self.target_mean = 0
        self.target_stddev = 0
        # normalization fields
        self.feature_max = 0
        self.feature_min = 0
        self.target_max = 0
        self.target_min = 0
        # whether or not to normalize targets
        self.normalize_targets = not self.targets.name in DONTNORMALIZE and NORMALIZE_TARGETS

    # execute the desired transform on the data in the dataset
    def transform(self):
        if MINMAX:
            self.minmax_scale(self.normalize_targets)
        elif STANDARD:
            self.standard_score(self.normalize_targets)
        else:
            return

    # z = (x - mean)/stddev for all x in each column
    # normalizes data to have mean of 0 and stddev of 1
    def standard_score(self, normalize_targets=False):
        for col in self.features.columns:
           self.feature_mean = self.features.loc[:,col].mean()
           self.feature_stddev = self.features.loc[:,col].std()
           z = []
           for i in range(len(self.features)):
               x = self.features[col].iloc[i]
               z.append(round((x - self.feature_mean)/self.feature_stddev, 2))
           self.features.loc[:, col] = z
        self.feature_tensor = torch.from_numpy(np.array(self.features)).float()
        """
        TARGET NORMALIZATION
        """
        if self.normalize_targets:
            self.target_mean = self .targets.mean()
            self.target_stddev = self.targets.std()
            z = []
            for i in range(len(self.targets)):
                x = self.targets.iloc[i]
                z.append(round((x - self.target_mean)/self.target_stddev, 2))
            self.targets = pd.Series(z, index=self.targets.index.values, name=self.targets.name)
            self.target_tensor = torch.from_numpy(np.array(self.targets)).float()

    def minmax_scale(self, normalize_targets=False):
        #normalize features
        for col in self.features.columns:
           self.feature_max = self.features.loc[:,col].max()
           self.feature_min = self.features.loc[:,col].min()
           x_diff = self.feature_max - self.feature_min
           z = []
           for i in range(len(self.features)):
               x = self.features[col].iloc[i]
               z.append(round((x - self.feature_min)/x_diff, 2))
           self.features.loc[:, col] = z
        # convert to PyTorch tensor
        self.feature_tensor = torch.from_numpy(np.array(self.features)).float()
        """
        TARGET NORMALIZATION
        """
        if self.normalize_targets:
            self.target_max = self.targets.max()
            self.target_min = self.targets.min()
            x_diff = self.target_max - self.target_min
            z = []
            for i in range(len(self.targets)):
                   x = self.targets.iloc[i]
                   z.append(round((x - self.target_min)/x_diff, 2))
            self.targets = pd.Series(z, index=self.targets.index.values, name=self.targets.name)
            self.target_tensor = torch.from_numpy(np.array(self.targets)).float()

    # invert the transform originally done on the data or do nothing if the data was never
    # initially scaled
    def inverse_transform(self, data):
        if not self.normalize_targets:
            return data
        elif MINMAX:
            return self.inverse_minmax_scale(data)
        elif STANDARD:
            return self.inverse_standard_score(data)
        else:
            return data

    # inverse the minmax scale operation done on the the target
    # and use the same inverse transform on the output data from the model
    def inverse_minmax_scale(self, data):
        x_diff = self.target_max - self.target_min
        z_data, z_target = [], []
        for i in range(len(self.targets)):
            x = self.targets.iloc[i]
            z_target.append(round(x*x_diff+self.target_min, 2))
            z_data.append(round(data[i]*x_diff+self.target_min, 2))
        self.targets = pd.Series(z_target, index=self.targets.index.values, name=self.targets.name)
        # convert to PyTorch tensor
        self.target_tensor = torch.from_numpy(np.array(self.targets)).float()

        return z_data

    # inverse the standard score operation done on the the target
    # and use the same inverse transform on the output data from the model
    def inverse_standard_score(self, data):
        z_data, z_target = [], []
        for i in range(len(self.targets)):
            x = self.targets.iloc[i]
            z_target.append(round(x*self.target_stddev + self.target_mean, 2))
            z_data.append(data[i]*self.target_stddev + self.target_mean)
        self.targets = pd.Series(z_target, index=self.targets.index.values, name=self.targets.name)
        self.target_tensor = torch.from_numpy(np.array(self.targets)).float()

        return z_data


    # return the item at the given index
    def __getitem__(self, index):
        x = self.feature_tensor[index]
        y = self.target_tensor[index]

        return x, y
        #return self.feature_tensor, self.target_tensor

    def __len__(self):
        return len(self.targets)

    # change the string representation to display the features and targets
    # dataframes
    def __str__(self):
        return "Features DF:\n{}\nFeatures Tensor:\n{}".format(self.features, self.feature_tensor) \
                    + "\nTargets DF:\n{}\nTargets Tensor:\n{}".format(self.targets, self.target_tensor)
