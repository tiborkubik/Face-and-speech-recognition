# # # # # # # # # #
# @title Voice classifier for SUR/2019L course using Gaussian Mixture Model
#
# @brief This code classifies speaker of .wav files
#
# @author Tibor Kubik (xkubik34@stud.fit.vutbr.cz)
#
# @author Andrej Jezik (xjezik03@stud.fit.vutbr.cz)
# # # # # # # # # #

import os
from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

# Train and evaluation paths for persons. Change for your need!
TRAIN_DIR = "../data/train"
EVAL_DIR = "../data/eval"
N_OF_CLASSES = 2            # edit this according to the number of training classes.

GAUSS_PER_PERSON = 16
TRAINING_ITERATIONS = 64

trainClasses = {}

# Specifing parameters for training
meanValues = {}
covarMatrices = {}
weights = {}


# Function loads all train data in separate classes and saves it into a dictionary
#
# @return dictionary, where key is name of class and value contains all features of all recordings for given class
def loadTrainData():
    filenames= os.listdir(TRAIN_DIR)

    # Recordings loading
    print("STEP1: Loading training data into internal structure")

    for filename in filenames:
        classRecords = wav16khz2mfcc(TRAIN_DIR + "/" + filename).values()
        trainClasses[filename] = classRecords
        trainClasses[filename] = np.vstack(classRecords)

    print("STEP1 Done: Training data loaded. Number of trainig classes: ", len(trainClasses), "\n")

    return trainClasses


# Function calculates mean values, covariance matrices and gauss weights for each class and returns them in separate dicitonaries
#
# @param trainClasses dictionary of all classes with its corresponding feature matrices
#
# @return 3 dictionaries, in each there are mean Values, Covariance matrices and gauss weights for every single class
def getTrainingParams(trainClasses):
    print("STEP2: Calculating mean values, covariance matrices and gauss weights for each class.")

    for singleClass in trainClasses:
        meanValues[singleClass] = (trainClasses[singleClass][randint(1, len(trainClasses[singleClass]), GAUSS_PER_PERSON)])
        covarMatrices[singleClass] = ([np.var(trainClasses[singleClass], axis=0)] * GAUSS_PER_PERSON)
        weights[singleClass] = np.ones(GAUSS_PER_PERSON) / GAUSS_PER_PERSON;

    print("STEP2 Done: All information for training for all classes are calculated.\n")

    return meanValues, covarMatrices, weights


# Function trains GMM for every class
#
# @param trainClasses all classes contaning all traning data
#
# @param gaussWeights
#
# @param meanValsGauss
#
# @param covarMatrices
#
# @return modified values needed for classification
def trainGMMs(trainClasses, gaussWeights, meanValsGauss, covarMatrices):
    print("STEP3: Training all train data using GMM.")
    # In each iteraiton, we have to make one training step for each training class
    for iteration in range(TRAINING_ITERATIONS):
        print("Training iteration: ", iteration)

        # calculating new values
        for singleClass in trainClasses:
            weights[singleClass], meanValues[singleClass], covarMatrices[singleClass], TTL = train_gmm( trainClasses[singleClass],
                                                                                                        weights[singleClass],
                                                                                                        meanValues[singleClass],
                                                                                                        covarMatrices[singleClass])

    print("STEP3 Done: Training finished.\n")

    return weights, meanValues, covarMatrices

# Function loads all evaluation data
#
# @return dictionary containing evaluation data
def loadEvalData():
    print("STEP4: Loading evaluation data.")

    evalData = wav16khz2mfcc(EVAL_DIR)

    print("STEP4 Done: All evaluation data loaded and prepared for classification.\n")

    return evalData

# This function provides the classification of all evaluation data and writes result in file
def classification(evalData, trainClasses, weights, meanValues, covarMatrices):
    print("STEP5: Classification started.")

    file = open("audio_GMM.txt", "w")

    # For every person to evaluate, calculate the sum of LLs for evaluation data
    for evalPerson in evalData:
        llVals = {}
        name = evalPerson.split('/')[3]
        name = name.split('.')[0]
        print("Classifing person ", name)
        file.write(name)
        file.write(' ')

        for trainPerson in trainClasses:
            llVals[trainPerson] = sum(logpdf_gmm(evalData[evalPerson],
                                                        weights[trainPerson],
                                                        meanValues[trainPerson],
                                                        covarMatrices[trainPerson]))

        llNonTarget = llVals["non-target"]
        llTarget = llVals["target"]

        softScore = (llTarget + np.log(0.5)) - (llNonTarget + np.log(0.5))

        file.write(str(softScore))

        file.write(' ')
        # Hard decision
        if(softScore > 500):
            file.write('1')
        else:
            file.write('0')

        file.write('\n')

    file.close()
    print("STEP5 Done: Classification ended.\n")

    print("Classification is finished. Check file \'audio_GMM.txt\' for the results.")

# main function
def main():
    trainClasses = loadTrainData()

    meanValues, covarMatrices, weights = getTrainingParams(trainClasses)

    weights, meanValues, covarMatrices = trainGMMs(trainClasses,weights, meanValues, covarMatrices)

    evalData = loadEvalData()

    classification(evalData, trainClasses, weights, meanValues, covarMatrices)


if __name__ == "__main__":
	main()
