"""
    Constants to be used throughout the project
"""

# root of the project directory
ROOT = "/home/zach/Documents/Coding/AIFinalProject/FF_Data/"
# which files to read in from
FILES = ["1992.csv", "1993.csv", "1994.csv",
          "1995.csv", "1996.csv", "1997.csv", "1998.csv", "1999.csv", "2000.csv", "2001.csv", "2002.csv",
          "2003.csv", "2004.csv", "2005.csv", "2006.csv", "2007.csv", "2008.csv", "2009.csv", "2010.csv",
           "2011.csv",  "2012.csv",  "2013.csv",  "2014.csv",  "2015.csv", "2016.csv", "2017.csv", "2018.csv",
           "2019.csv",  "2020.csv"]
# which columns input to ignore
SKIP_COLS = [1, 2, 5, 25, 28, 29, 30]
# how many rows to skip at the top of the csv
SKIP_ROWS = 1
# every column name including those not used
COMPLETENAMES = ["Player", 'Tm', 'FantPos', "Age", "G", 'GS', "Cmp", "QBAtt", "QBYds", "QBY/A", "QBTD",
         "Int", "RBAtt", "RBYds", "Y/A", "RBTD", "Tgt", "Rec", "WRYds", "Y/R", "WRTD",
         "Fmb", "FL", "TOTTD", "2PM", '2PP', "FantPt", "PPRPt", 'DKPt', 'FDPt', 'VBD', "PosRank", "QBTD/G",
         "Cmp/G", "QBAtt/G", "QBYds/G", "Int/G", "RBAtt/G", "RBYds/G", "Rec/G",
         "WRYds/G", "WRTD/G", "Fmb/G", "TOTTD/G", "2PM/G", "FantPt/G", "PPR/G",
         'QBPt/G', 'RBPt/G', 'WRPt/G', 'WRPPRPt/G', 'QB', 'RB', 'WR', 'TE', '3Cat', '6Cat']
# columns in the CSV generated by running configure_csvs()
NAMES = ["Player", "Age", "G", "Cmp", "QBAtt", "QBYds", "QBY/A", "QBTD",
         "Int", "RBAtt", "RBYds", "Y/A", "RBTD", "Tgt", "Rec", "WRYds", "Y/R", "WRTD",
         "Fmb", "FL", "TOTTD", "2PM", "FantPt", "PPRPt", "PosRank", "QBTD/G",
         "Cmp/G", "QBAtt/G", "QBYds/G", "Int/G", "RBAtt/G", "RBYds/G", "Rec/G",
         "WRYds/G", "WRTD/G", "Fmb/G", "TOTTD/G", "2PM/G", "FantPt/G", "PPR/G",
         'QBPt/G', 'RBPt/G', 'WRPt/G', 'WRPPRPt/G', 'QB', 'RB', 'WR', 'TE', '3Cat', '6Cat']
# columns provided in orignal CSVs  minus the 'Rk' category
NAMES2 =  ["Player",  "Tm", "FantPos", "Age", "G", "GS", "Cmp", "QBAtt", "QBYds", "QBTD",
         "Int", "RBAtt", "RBYds", "Y/A", "RBTD", "Tgt", "Rec", "WRYds", "Y/R", "WRTD",
         "Fmb", "FL", "TOTTD", "2PM", "2PP", "FantPt", "PPRPt", "DKPt", "FDPt",  "VBD", "PosRank"]
# columns to use for network looking exclusively at stats per game, age, and games played
PGNAMES = ["Age", "G", "QBTD/G", "Cmp/G", "QBAtt/G", "QBYds/G", "Int/G", "RBAtt/G", "RBYds/G", "Rec/G",
                         "WRYds/G", "WRTD/G", "Fmb/G", "TOTTD/G", "2PM/G", "FantPt/G", "PPR/G"]
# columns to use for network using stats per game, age, games played and yards per play (Y/A, Y/R...)
PGPPNAMES = ["Age", "G", "QBY/A", "Y/A", "Y/R", "QBTD/G", "Cmp/G", "QBAtt/G", "QBYds/G",
                             "Int/G", "RBAtt/G", "RBYds/G", "Rec/G", "WRYds/G", "WRTD/G", "Fmb/G", "TOTTD/G",
                             "2PM/G", "FantPt/G", "PPR/G"]
# columns to use for network using whole season stats and yard per play
PSNAMES = ["Age", "G", "Cmp", "QBAtt", "QBYds", "QBY/A", "QBTD", 
                         "Int", "RBAtt", "RBYds", "Y/A", "RBTD", "Tgt", "Rec", "WRYds", "Y/R", "WRTD",
                         "Fmb", "FL", "TOTTD", "FantPt", "PPRPt", "PosRank"]
# columns to use for the games played network
GPNAMES = ["Age", "G", "QBAtt", "RBAtt", "Tgt", "Rec",  "QBAtt/G", "Int/G", "RBAtt/G", "Rec/G", "Fmb/G",
              "QB", 'RB', 'WR', 'TE']
# columns to use for the QB network
QBNAMES = ["QBY/A", "QBTD", "Int", "QBTD/G", "Cmp/G", "QBAtt/G",
                          "QBYds/G", "Int/G", "QBPt/G", 'QB']
# columns to use for the RB network
RBNAMES = ["RBAtt", "RBYds", "Y/A", "RBTD", "Fmb", "FL", 
                         "RBAtt/G", "RBYds/G", "Fmb/G", "RBPt/G", 'RB', 'QB', 'WR', 'TE']
# columns to use for the WR network
WRNAMES = ["Tgt", "Y/R", "WRTD", "Rec/G", "WRYds/G",
                           "WRTD/G", "TOTTD/G", "WRPt/G", 'QB']#, 'WR', 'RB', 'TE'], "Rec", "WRYds",
# columns to use for the overall network
OVNAMES = ['Age', "G", "Cmp", "QBAtt", "QBYds", "QBY/A", "QBTD",
         "Int", "RBAtt", "RBYds", "Y/A", "RBTD", "Tgt", "Rec", "WRYds", "Y/R", "WRTD",
         "Fmb", "FL", "TOTTD", "2PM", "FantPt", "PosRank", "QBTD/G",
         "Cmp/G", "QBAtt/G", "QBYds/G", "Int/G", "RBAtt/G", "RBYds/G", "Rec/G",
         "WRYds/G", "WRTD/G", "Fmb/G", "TOTTD/G", "2PM/G", "FantPt/G", 
         'QBPt/G', 'RBPt/G', 'WRPt/G', 'QB', 'RB', 'WR', 'TE', '3Cat', '6Cat']
# list of columns to use for all features
ALLNAMES = [OVNAMES, OVNAMES]
# different datasets for different targets and different subsets of features
#TARGETS = ['G', 'QBPt/G', 'RBPt/G', 'WRPt/G', 'FantPt/G','FantPt', '3Cat', '6Cat', 'FantPt']
#FEATURES = ['GP', 'QBPt', 'RBPt', 'WRPt', 'Overall', 'Pred', '3Cat', '6Cat', 'Bagging']
#TARGETS = ['3Cat', 'FantPt']
#FEATURES = ['3Cat', 'Stacked']
TARGETS = ['FantPt']
FEATURES = ['Regression']
# how many trials to run on a network
TRIALS = 5
# should we normalize input data?
STANDARD = False
MINMAX = True
# should we normalize the targets?
NORMALIZE_TARGETS = False
# never normalize these columns
DONTNORMALIZE = ['QB', 'RB', 'WR', 'TE', 'FantPt', '3Cat', '6Cat']
# different learning rates to try for each type of loss
#RATES = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.00001]
RATES = [0.01]
# thresholds
THRESHOLDS = [3.5, .7, .9, 1.8, 2.6, 41]
# determines how to determine fantasy point output for classifier models
INTERVALS = [0, 117, 300]
# number of rows to read in from the csv file
NROWS = 250
