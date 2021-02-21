import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
# local imports
import regression_networks
from regression_networks import *
import classification_networks
from classification_networks import *
import constants 
from constants import *
import get_FF_data
from get_FF_data import *


# TODO: Make a class with the functionality of choosing a model, training it, testing it and saving it
class ModelMaker():
    def __init__():
        pass
    # train the given model on the data in train_loader, run
    # training for epochs iterations and plot the loss
    # versus epochs on a matplot lib plot with the given title
    def train(model, train_loader, epochs, title, model_type):
        criterion = model.criterion
        optimizer = model.optimizer
        epoch_count , error = [], []
        for e in range(epochs):
            epoch_loss = 0;
            for features, target in train_loader:
                #print("Features: {}".format(features))
                #print("Target: {}".format(target))
                if model_type == "regression":
                    # for regression networks
                    target = torch.unsqueeze(target, 1) # format target to same size as model output
                elif model_type == "classification":
                    # for classification networks
                    target = target.long()
                else:
                    print("Unrecognized model type {}, edit model_type input to train() and try again.".format(model_type))
                    return
                # Training pass
                optimizer.zero_grad()               # zero gradient buffers
                output = model(features)         # run the data through the model
                #print("Output: {}".format(output))
                loss = criterion(output, target) # calculate the loss
                loss.backward()                           # backpropogate
                optimizer.step()
                epoch_loss +=loss.item()
            else:
                epoch_count.append(e)
                #print("Loss: {}".format(epoch_loss/len(train_loader)))

        #plot_data(epoch_count, error, "Epochs", "Loss Function", title)
        return model


    # train all of the models in the ensemble with Learning Rate alpha for
    # epochs iterations
    def train_ensemble(train_loaders, validate_loaders, epochs):
        """
        # games played model
        while(True):
            gp_model = GamesPlayedModel()
            train(gp_model, train_loaders['GP'], epochs, "GP")
            verr, verr2 = test(gp_model, validate_loaders['GP'], "GP")
            if verr <= THRESHOLDS[0]:
                break

        print("GP: Average Validate Error: {}, Average Error Squared: {}".format(verr, verr2))
        
        while(True):
            qb_model = QBModel()
            train(qb_model, train_loaders['QBPt'], epochs, "QB")
            verr, verr2 = test(qb_model, validate_loaders['QBPt'], "QB")
            if verr <= THRESHOLDS[1]:
                break

        print("QB: Average Validate Error: {}, Average Error Squared: {}".format(verr, verr2))

        while(True):
            rb_model = RBModel()
            train(rb_model, train_loaders['RBPt'], epochs, "RB")
            verr, verr2 = test(rb_model, validate_loaders['RBPt'], "RB")
            if verr <= THRESHOLDS[2]:
                break

        print("RB: Average Validate Error: {}, Average Error Squared: {}".format(verr, verr2))

        while(True):
            wr_model = WRModel()
            train(wr_model, train_loaders['WRPt'], epochs, "WR")
            verr, verr2 = test(wr_model, validate_loaders['WRPt'], "WR")
            if verr <= THRESHOLDS[3]:
                break

        print("WR: Average Validate Error: {}, Average Error Squared: {}".format(verr, verr))

        while(True):
            ov_model = OverallModel()
            train(ov_model, train_loaders['Overall'], epochs, "FantPt/G")
            verr, verr2 = test(ov_model, validate_loaders['Overall'], "FantPt/G")
            if verr <= THRESHOLDS[4]:
                break

        print("Overall: Average Validate Error: {}, Average Error Squared: {}".format(verr, verr2))

        while(True):
            sp_model = SinglePredictorModel()
            train(sp_model, train_loaders['Pred'], epochs, "FantPt")
            verr, verr2 = test(sp_model, validate_loaders['Pred'], "FantPt")
            if verr <= THRESHOLDS[5]:
                break

        print("Single Prediction: Average Validate Error: {}, Average Error Squared: {}".format(verr, verr2))
        """
        verr = 1
        while(verr > 0.36):
            classification_model = ClassificationModel()
            print(classification_model)
            train(classification_model, train_loaders['3Cat'], epochs, "3 Category Classification Model", "classification")
            verr, verr2 = test(classification_model, validate_loaders['3Cat'],"3 Category Classification on Validation Data", "classification")

        return classification_model


    # test a pretrained model's accuracy on
    # a test or validation set
    def test(model, test_loader, title, model_type, verbose=False):
        error, error2 = 0, 0
        predictions = []
        # iterate over all of the validation set and keep
        # a running total of the difference between the
        # model's prediction and actual value
        for features, target in test_loader:
            prediction = model(features);
            if model_type == "regression":
                predictions.append(prediction.item())
            elif model_type == "classification":
                predictions.append(torch.argmax(prediction).item())
                """
                prediction = scale_probabilities(prediction.tolist()[0])
                predicted_score = 0
                for i in range(len(prediction)):
                    predicted_score += INTERVALS[i]*prediction[i]
                predictions.append(predicted_score)
                """
            else:
                print("Unrecognized model type {}, edit model_type input to test() and try again.".format(model_type))
                return
                
        predictions = test_loader.dataset.inverse_transform(predictions)
        i, num_correct = 0, 0
        for features, target in test_loader:
            prediction = predictions[i]
            err = abs(target.item() - prediction)
            error += err
            error2 += err**2
            if err == 0:
                num_correct+=1
            i+=1
            if verbose:
                print("Prediction: {} Actual: {}".format(prediction, target.item()))
        # average error per player
        error_per_player = round(error/len(test_loader), 3)
        error2_per_player = round(error2/len(test_loader), 3)
        percent_correct = round(num_correct/len(test_loader), 3) * 100
        print("######### {} #########".format(title))
        print("Error per player: {}".format(error_per_player))
        print("Error squared per player: {}".format(error2_per_player))
        print("Percent correct answers: {}".format(percent_correct))

        test_loader.dataset.transform()
        return error_per_player, error2_per_player


    # make a matplot lib plot with the given x and y data
    # arrays and with the given axis lables and title
    def plot_data(x1, y1, xlabel, ylabel, title):
        plt.plot(x1, y1, 'ro-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


    # scale class probabilities to be between 0 and 1 if they aren't already
    def scale_probabilities(probs):
        # sum of probabilities should add to 1, see if it does
        total = sum(probs)
        # if not scale each probability so they do add to 1
        if not sum == 1:
            for i in range(len(probs)):
                probs[i] = round(probs[i]/total, 2)
        return probs


    # train and use the models 
    def test_():

        # how many iterations to train the networks on
        epochs = 150
        # get the data already formatted to be easily accessible and
        # to be able to be passed directly into the train function
        train_loaders, validate_loaders, test_loaders, train_loaders_test= get_FF_data()
        # train the ensemble of learners
        #classification_model = train_ensemble(train_loaders, validate_loaders, epochs)
        # analyze the bagging model using the trained ensemble
        analyze_model(train_loaders, validate_loaders, test_loaders, train_loaders_test)

        #return gp_model, qb_model, rb_model, wr_model, ov_model, train_loaders, validate_loaders, test_loaders, train_loaders_test
        return

    # analyze a model
    def analyze_model(train_loaders, validate_loaders, test_loaders, train_loaders_test,classification_model=None,epochs=150):
        # errors and errors squared per learning rates
        trate_error = []
        trate_error2 = []
        vrate_error = []
        vrate_error2 = []
        #frate_error = []
        #frate_error2 = []
        # labels and titles for the final plot
        title_t = "QB Model on Training Set"
        title_v = "QB Model on Validation Set"
        # test bagging model over all learning rates in RATES (constants.py)
        for alpha in RATES:
            # boolean to determine whether to accept and plot the results
            accept = False
            # while we have not yet had at least 1 good trial on the bagging
            # model for this learning rate, alpha
            while not accept:
                # how many bad runs
                bad_runs = 0
                # what average error constitutes a bad run
                threshold = 50
                # hold error and error squared per trial on validation set
                verror = []
                verror2 = []
                # hold average error and error squared per trial on train set
                terror = []
                terror2 = []
                # hold average error and error squared per trial on test set
                #ferror = []
                #ferror2 = []
                # average error and error squared over all trials on validation set
                vavg_error = 0
                vavg_error2 = 0
                # average error and error squared over all trials on train set
                tavg_error = 0
                tavg_error2 = 0
                # average error and error squared over all trials on test set
                #favg_error = 0
                #favg_error2 = 0
                # train and test the bagging model for TRIALS (constants.py) trials
                for i in range(TRIALS):
                    """
                    # create the Bagging network
                    bagging_model = GPModel(nn.SmoothL1Loss(), alpha, gp_model, qb_model, rb_model, wr_model, ov_model, sp_model)
                    # train the Bagging network
                    train(bagging_model, train_loaders['Bagging'], epochs, "Ensemble")
                    # test accuracy on training data
                    terr, terr2 = test(bagging_model, train_loaders_test['Bagging'], "Ensemble on Training Set")
                    # test accuracy on validation data
                    verr, verr2 = test(bagging_model, validate_loaders['Bagging'], "Ensemble on Validation Set", True)
                    # test accuracy on test data
                    #ferr, ferr2 = test(bagging_model, test_loaders['Bagging'], "Ensemble on Test Set")
                    """
                    # create the stacked network
                    model = SinglePredictorModel()
                    #print(train_loaders["3Cat"].dataset)
                    # train the Bagging network
                    #print(train_loaders['3Cat'].dataset)
                    train(model, train_loaders['Regression'], epochs, "Stacked Model", "regression")
                    # test accuracy on training data
                    terr, terr2 = test(model, train_loaders_test['Regression'], "Stacked Model on Training Set", "regression")
                    # test accuracy on validation data
                    verr, verr2 = test(model, validate_loaders['Regression'], "Stacked Model on Validation Set", "regression", True)
                    # test accuracy on test data
                    #ferr, ferr2 = test(model, test_loaders['Bagging'], "Ensemble on Test Set")

                    # append errors to list of errors per trial
                    terror.append(terr)
                    terror2.append(terr2)
                    verror.append(verr)
                    verror2.append(verr2)
                    #ferror.append(ferr)
                    #ferror2.append(ferr2)
                    # was this a bad run?
                    if verr > threshold:
                        bad_runs+=1
                        continue
                    
                    # increment running total error
                    vavg_error += verror[i]
                    vavg_error2 += verror2[i]
                    tavg_error += terror[i]
                    tavg_error2 += terror2[i]
                    #favg_error += ferror[i]
                    #favg_error2 += ferror2[i]
                # if no good runs occur after TRIALS trials, try again
                if TRIALS - bad_runs == 0:
                    print("No good runs encountered, trying again")
                else:
                    accept = True

            # divide running total average errors and errors squared on good
            # trials by the amount of good trials completed to get average per trial
            vavg_error = round(vavg_error/(TRIALS - bad_runs), 3)
            vavg_error2 = round(vavg_error2/(TRIALS - bad_runs), 3)
            tavg_error = round(tavg_error/(TRIALS - bad_runs), 3)
            tavg_error2 = round(tavg_error2/(TRIALS - bad_runs), 3)
            #favg_error = round(favg_error/(TRIALS - bad_runs), 3)
            #favg_error2 = round(favg_error2/(TRIALS - bad_runs), 3)
            # print out the error and error squared for each trial as well as the averages on good runs
            print("######### Learning Rate = {} #########".format(alpha))
            for i in range(TRIALS):
                print("Train {} Error: {}, Error Squared: {}".format(i+1, terror[i], terror2[i]))
                print("Trial {} Error: {}, Error Squared: {}".format(i+1, verror[i], verror2[i]))
                #print("Test {} Error: {}, Error Squared: {}".format(i+1, ferror[i], ferror2[i]))
            print("Average Train Error: {}, Average Error Squared: {}".format(tavg_error, tavg_error2))
            print("Average Validate Error: {}, Average Error Squared: {}".format(vavg_error, vavg_error2))
            #print("Average Test Error: {}, Average Error Squared: {}".format(favg_error, favg_error2))
            # append the good run averages to the lists of averages
            trate_error.append(tavg_error)
            trate_error2.append(tavg_error2)
            vrate_error.append(vavg_error)
            vrate_error2.append(vavg_error2)

        # comment out if you want plots of Learning Rate Vs Error and Learning Rate Vs Error Squared
        return

        """
            PLOT LEARNING RATE VS AVERAGE ERROR AND SQUARED ERROR FOR THE MODEL
        """

        # plot rates vs avg error and avg squared error
        plot_data(RATES, trate_error, "Learning Rate", "Average Error", "RB Model: Learning Rate vs Average Error for Smooth L1 Loss Train Set")
        plot_data(RATES, trate_error2, "Learning Rate", "Average Error Squared", "RB Model: Learning Rate vs Average Squared"+
                                                                                                                                        " Error for Smooth L1 Loss Train Set")
        plot_data(RATES, vrate_error, "Learning Rate", "Average Error", "RB Model: Learning Rate vs Average Error for Smooth L1 Loss Validation Set")
        plot_data(RATES, vrate_error2, "Learning Rate", "Average Error Squared", "RB Model: Learning Rate vs Average Squared" +
                                                                                                                                        " Error for Smooth L1 Loss Validation Set")
        #plot_data(RATES, frate_error, "Learning Rate", "Average Error", "Bagging Model: Learning Rate vs Average Error for Smooth L1 Loss Test Set")
        #plot_data(RATES, frate_error2, "Learning Rate", "Average Error Squared", "Bagging Model: Learning Rate vs Average Squared" +
        #                                                                                                                                " Error for Smooth L1 Loss Test Set")

