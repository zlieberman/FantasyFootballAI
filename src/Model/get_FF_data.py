import pandas as pd
import torch
from torch.utils.data import DataLoader
import copy
# import needed files from project
import regression_networks
from regression_networks import *
import constants 
from constants import *
import FFDataset
from FFDataset import *

# read in data from the csvs specified in files with the file
# path root. Places the data into dataframes using names
# as the columns names and skipping the specified columns
# and rows
def read_csvs(files, root, names, skipcols, skiprows):
    cols = []
    for i in range(len(names) + len(skipcols)):
        if not i in skipcols:
            cols.append(i)
            
    data = []
    for filename in files:
        filename = root + filename
        # reading the CSV file  
        df = pd.read_csv(filename, names=names, usecols=cols, skiprows=skiprows, nrows=NROWS)
        data.append(df)

    return data

# make sure player names are consistent from season to season
# and that players that don't appear in consecutive seasons
# are not passed into the network as they would either be
# missing features or targets
def clean_data(data):
    # edit the player names to be consistent from season to season
    for df in data:
        df.fillna(0, inplace=True)   # replace all NaN values with 0
        #df["Player"] = df.apply(lambda x: x["Player"].split("\\")[1], axis=1) # split player names at delimiter
        df.set_index("Player", inplace=True) # set index to the player name column

    targets = []
    configured_data = copy.deepcopy(data)

    # this loop ensures that the players in the target used for each
    # season match the players in the feature list for that season
    for i in range(len(data) - 1):
        for name in data[i].index.values:
            if not name in data[i+1].index.values:
                configured_data[i].drop(name, inplace=True)
        targets.append(copy.deepcopy(data[i+1][TARGETS]))
        for name in targets[i].index.values:
            if not name in configured_data[i].index.values:
                targets[i].drop(name, inplace=True)
        targets[i] = targets[i].reindex(configured_data[i].index.values.tolist())

    return configured_data[0:len(data)-1], targets

# format the data into dictionaries of PyTorch DataLoaders
# used to train the network. Dictionary is indexed by the
# different sets of features being used. Inside of each set
# of features are different DataLoaders for each of the
# different types of targets available
def format_data(train_features_df, train_targets_df, validate_features_df, validate_targets_df, test_features_df, test_targets_df):
    # lists to hold the different types of features
    train_features = []
    validate_features = []
    test_features = []
    for names in ALLNAMES:
        train_features.append(train_features_df[names])
        validate_features.append(validate_features_df[names])
        test_features.append(test_features_df[names])
        
    # four different kinds of targets: fantasy points, fantasy points per game,
    #                                                      overall rank, position rank
    train_targets = []
    validate_targets = []
    test_targets = []
    for category in TARGETS:
        train_targets.append(train_targets_df[category])
        validate_targets.append(validate_targets_df[category])
        test_targets.append(test_targets_df[category])

    # convert the data to PyTorch tensors to be used in the neural network
    # this is a dictionary mapping names of the different feature types to dictionaries
    # mapping the different target names per feature to the DataLoader for that
    # feature/target pair
    train_loaders = dict((f, []) for f in FEATURES)
    train_loaders_test = dict((f, []) for f in FEATURES)
    validate_loaders = dict((f, []) for f in FEATURES)
    test_loaders = dict((f, []) for f in FEATURES)
    # arrays to store preprocessed data for the bagging model
    bagging_train = []
    bagging_validate = []
    bagging_test = []
    # tensor datasets for each of the types of targets for each of the types of features
    for j in range(len(FEATURES)):
        # create the dataset converting the dataframes first into numpy arrays, then into torch tensors
        # of float values
        train_dataset = FFDataset(train_features[j], train_targets[j])
        validate_dataset = FFDataset(validate_features[j], validate_targets[j])
        test_dataset = FFDataset(test_features[j], test_targets[j])
        # if we want to normalize the data call the transform method
        # in the custom FFDataset class
        train_dataset.transform()
        validate_dataset.transform()
        test_dataset.transform()
        # append (potentially) normalized features to the ensemble feature arrays
        bagging_train.append(train_dataset.features)
        bagging_validate.append(validate_dataset.features)
        bagging_test.append(test_dataset.features)
        # take the datasets and convert them to PyTorch data loaders and place them in the appropriate slot
        # in the dictionary
        train_loaders[FEATURES[j]] = DataLoader(train_dataset, batch_size=10, shuffle=True)
        train_loaders_test[FEATURES[j]] = DataLoader(train_dataset)
        validate_loaders[FEATURES[j]] = DataLoader(validate_dataset)
        test_loaders[FEATURES[j]] = DataLoader(test_dataset)

    """
    # format the data for the ensemble bagging model
    train_dataset = FFDataset(pd.concat(bagging_train, axis=1), train_targets[len(TARGETS)-1])
    validate_dataset = FFDataset(pd.concat(bagging_validate, axis=1), validate_targets[len(TARGETS)-1])
    teset_dataset = FFDataset(pd.concat(bagging_test, axis=1), test_targets[len(TARGETS)-1])
    #initialize the data loaders
    train_loaders["Bagging"] = DataLoader(train_dataset, batch_size=10)
    train_loaders_test["Bagging"] = DataLoader(train_dataset)
    validate_loaders["Bagging"] = DataLoader(validate_dataset)
    test_loaders["Bagging"] = DataLoader(test_dataset)
    """

    return train_loaders, validate_loaders, test_loaders, train_loaders_test

# grab the raw dataframes from reading the csv and configure them
# to this specific application
def get_FF_data():
    
    # which files are being used
    files = FILES[8:len(FILES)-1]

    # read the csvs into a list of pandas dataframes
    data = read_csvs(files, ROOT, NAMES, SKIP_COLS, SKIP_ROWS)

    """
    df = pd.concat(data[len(data)-11:len(data)-1])
    print("Data Mean for Last 10 Years: {}".format(df["FantPt"].mean()))
    print("Data Standard Deviation for Last 10 Years: {}".format(df["FantPt"].std()))
    """

    # clean up the data so it is usable for the neural network
    configured_data, targets = clean_data(data)

    # create dataframes for test, validation and training features and targets
    train_features_df = pd.concat(configured_data[0:len(targets)-7])
    train_targets_df = pd.concat(targets[0:len(targets)-7])
    train_targets_df = train_targets_df.T.drop_duplicates().T # remove duplicates
    validate_features_df = pd.concat(configured_data[len(targets)-6: len(targets)-3])
    validate_targets_df = pd.concat(targets[len(targets)-6:len(targets)-3])
    validate_targets_df = validate_targets_df.T.drop_duplicates().T
    test_features_df = pd.concat(configured_data[len(targets)-2:len(targets)])
    test_targets_df = pd.concat(targets[len(targets)-2:len(targets)-1])
    test_targets_df = test_targets_df.T.drop_duplicates().T

    """
    # print the data to the user
    print("######### Train Data #########")
    print("features length: {}, targets length: {}".format(len(train_features_df), len(train_targets_df)))
    print(train_features_df)
    print(train_targets_df)
    print("######### Validate Data #########")
    print("features length: {}, targets length: {}".format(len(validate_features_df), len(validate_targets_df)))
    print(validate_features_df)
    print(validate_targets_df)
    print("######### Test Data #########")
    print("features length: {}, targets length: {}".format(len(test_features_df), len(test_targets_df)))
    print(test_features_df)
    print(test_targets_df)
    """
    train_loaders, validate_loaders, test_loaders, train_loaders_test = format_data(train_features_df,
                                                                                                             train_targets_df,
                                                                                                             validate_features_df,
                                                                                                             validate_targets_df,
                                                                                                             test_features_df,
                                                                                                             test_targets_df)

    return train_loaders, validate_loaders, test_loaders, train_loaders_test
