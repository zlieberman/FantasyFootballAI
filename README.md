Overview of code, organized by file:

Main Files:

get_FF_data.py:

Summary: 

- this file serves the purpose of reading in the data from the csv files and converting it into a usable form, in this case a dictionary of PyTorch DataLoader objects used in training and testing. The dictionary is indexed by feature types so that the data used for each individual model can be compiled and accessed each run as opposed to having to modify this file any time a different model was to be tested.

Functions:

read_csvs():

Arguments:

- files: list of files without the file path (list of string)
- root: file path for all of the files (string)
- names: what to title the columns of the dataframe to be created (list of string)
- skipcols: what columns to ignore (list of integers)
- skiprows: how many rows to ignore at the top of the csv (integer)

Output:

- returns a list of Pandas DataFrame objects containing the data from the csvs in files

clean_data():

Arguments:

- data: list of Pandas DataFrame objects returned by read_csvs()

Output:

- returns a two lists of Pandas Dataframe objects. The Series list is the list of targets and the Dataframe list is the list of features. This function replaces all 'nan' values in data with 0, sets the 'Player' name column to be the index of both the features and targets and removes players who do not appear in consecutive seasons.

format_data():

Arguments:

- train_features_df: Pandas DataFrame containing all of the training features to be used
- train_targets_df: Pandas DataFrame containing all categories of targets to be used for the different feature models' training sets
- validate_features_df: Pandas DataFrame containing all of the validation set features to be used
- validate_targets_df: Pandas DataFrame containing all categories of targets to be used for the different feature models' validation sets
- test_features_df: Pandas DataFrame containing all of the test set features to be used
- test_targets_df: Pandas DataFrame containing all categories of targets to be used for the different feature models' test sets

Output:

- train_loaders: Dictionary of PyTorch DataLoaders indexed by feature model to be used to train each individual feature model
- validate_loaders: Dictionary of PyTorch DataLoaders indexed by feature model containing validation set data
- test_loaders: Dictionary of PyTorch DataLoaders indexed by feature model containing test set data
- train_loaders_test: Same as train_loaders except with a batch size of 1 (each call to the DataLoader only takes 1 as opposed to 10 rows) to be used to test accuracy on the training set

get_FF_data():

Arguments: None

Output:

- returns the output from format_data(). This is the main function of this file using the others as helpers.

networks.py

Summary:

- This file contains all of the classes needed to be defined for the code to run. This includes a custom PyTorch DataSet class and all of the models used throughout the project.

Classes:

FFDataset:

Inputs: 

- features: Pandas DataFrame of features to be used on the model
- targets:  Pandas Series of targets to be used on the model

Attributes:

- features
- targets
- feature_tensor: features converted to a PyTorch tensor of floats
- target_tensor: targets converted to a PyTorch tensor of floats

Methods:

- standard_score(): calculates the standard (normal) score of each column of the features DataFrame
- minmax_scale(): scales each column of the features DataFrame to a value between 0 and 1
- __getitem__(): returns the appropriate index of feature_tensor and target_tensor
- __len__(): returns the length of the input
- __str__(): sets the string representation of an FFDataset to pring out all four attributes in the order they are listed under Attributes

BaggingModel:

Inputs:

- gpmodel: pretrained GamesPlayedModel
- qbmodel: pretrained QBModel
- rbmodel: pretrained RBModel
- wrmodel: pretrained WRModel
- ovmodel: pretrained OverallModel
- lr: Learning Rate to be used in the optimizer
- criterion: Type of loss function to use

Attributes:

- gpmodel
- qbmodel
- rbmodel
- wrmodel
- lr
- criterion
- gp_idx: where gpmodel data ends
- qb_idx: where qbmodel data ends
- rb_idx: where rbmodel data ends
- wr_idx: where wrmodel data ends
- ov_idx: where ovmodel data ends 
- hidden: first hidden layer [5x2]
- output: output layer [2x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses bagging to predict player fantasy point totals for the following season
- uses standard normalization
- uses ReLU activation function

SinglePredictorModel:

Inputs:

- lr: Learning Rate to be used in the optimizer
- criterion: type of loss function to use

Attributes:

- lr
- criterion
- hidden: first hidden layer [44x22]
- hidden1: second hidden layer [22x11]
- output: output layer [11x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses a single network to predict player fantasy point totals for the following season
- uses ****** normalization
- uses ReLU activation function

OverallModel:

Inputs:

- lr: Learning rate to be used by the optimizer
- criterion: type of Loss function to use

Attributes:

- lr
- criterion
- hidden: first hidden layer [44x22]
- output: output layer [22x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses a single network to predict fantasy points per game for each player
- uses standard normalization
- uses ReLU activation function
- uses Smooth L1 (Huber) Loss function

GamesPlayedModel:

Inputs:

- lr: Learning rate to be used by the optimizer
- criterion: type of Loss function to use

Attributes:

- lr
- criterion
- hidden: first hidden layer [15x8]
- output: output layer [8x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses a single network to predict games played for each player
- uses standard normalization
- uses ReLU activation function
- uses MSE Loss function

QBModel:

Inputs:

- lr: Learning rate to be used by the optimizer
- criterion: type of Loss function to use

Attributes:

- lr
- criterion
- hidden: first hidden layer [10x5]
- output: output layer [5x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses a single network to predict passing fantasy points per game for each player
- uses standard normalization
- uses ReLU activation function
- uses L1 Loss function

RBModel:

Inputs:

- lr: Learning rate to be used by the optimizer
- criterion: type of Loss function to use

Attributes:

- lr
- criterion
- hidden: first hidden layer [14x7]
- output: output layer [7x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses a single network to predict rushing fantasy points per game for each player
- uses standard normalization
- uses ReLU activation function
- uses Smooth L1 Loss

WRModel:

Inputs:

- lr: Learning rate to be used by the optimizer
- criterion: type of Loss function to use

Attributes:

- lr
- criterion
- hidden: first hidden layer [9x5]
- output: output layer [5x1]
- optimizer: type of optimizer to be used, SGD here

Notes:

- uses a single network to predict receiving fantasy points per game for each player
- uses standard normalization
- uses ReLU activation function
- uses MSE Loss function

main.py:

Summary:

- This is where the main code is executed including training, testing and analysis of models.

Functions:

train():

Arguments:

- model: the model to be trained
- train_loader: PyTorch DataLoader containing the data to be used for training
- epochs: the number of full training runs to do on the model
- title: if desired the title can be used to plot the error vs epoch by uncommenting the last line of the function

Output:

- None, edits the model inplace. 

train_ensemble():

Arguments:

- alpha: learning rate to use for the models
- train_loaders: dictionary of PyTorch DataLoaders to be used by the individual learners in the ensemble for the training set
- validate_loaders: dictionary of PyTorch DataLoaders for the validation set
- epochs: number of full runs on the training data to do for all the models in the ensemble

Output:

- returns the trained models in the ensemble

test():

Arguments:

- model: the model to be tested
- test_loader: PyTorch DataLoader containing the data to be tested on the model
- title: description of the model being tested and displayed with the results to the user in the terminal
- verbose: boolean that defaults to False. If it is True the prediction and actual for each data point will be printed to the user

Output:

- returns the average error per player |prediction - actual|/num_players and the average error squared per player |prediction - actual|^2/num_players

plot_data():

Arguments:

- x1: x data
- y1: y data
- xlabel: label of the x-axis
- ylabel: label of the y-axis
- title: title of the plot

Output:

- returns nothing but creates a matplotlib graph of x1 vs y1 with the appropriate labels

analyze_model():

Arguments:

- train_loaders: dictionary of DataLoaders to be used for training
- validate_loaders: dictionary of DataLoaders to be used for validation set
- test_loaders: dictionary of DataLoaders to be used for testing
- train_loaders_test: train_loaders where each DataLoader has a batch size of 1 instead of 10
- epochs: number of full runs through the training data to do
- gp_model: used for bagging network, defaults to none
- qb_model: used for bagging network, defaults to none
- rb_model: used for bagging network, defaults to none
- wr_model: used for bagging network, defaults to none
- ov_model: used for bagging network, defaults to none
- sp_model: used for bagging network, defaults to none

Output:

- outputs nothing but plots the average error and average error squared versus learning rate for all rates in RATES (constants.py) for the model on the training and validation sets

main():

- main method of this file, calls all of the other methods in the desired order on the desired inputs.

constants.py:

Summary:

- contains all all-caps constants throughout the code. The meaning of all of the constants can be found in a comment the line above where it is defined. In order to try out different parameters for models on the code as well as number of trials, type of normalization and other attributes, some of these constants can be altered.

configure_FF_csvs.py:

- reads in the data from the raw csv and formats it in the desired way. Adds new categories using the ones provided in the orginal csv.
 



