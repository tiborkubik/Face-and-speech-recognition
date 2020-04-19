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
import sys
from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm
import matplotlib.pyplot as plt
import pprint
import numpy as np
from numpy.random import randint
from glob import glob
import math

# Train and evaluation paths for persons. Change for your need!
TRAIN_DIR = "data/train2"
EVAL_DIR = "data/eval2"
N_OF_CLASSES = 2            # edit this according to the number of training classes.

GAUSS_PER_PERSON = 8
TRAINING_ITERATIONS = 20


trainClasses = {}

# Specifing parameters for training
meanValsGauss = {}
covarMatrices = {}
gaussWeights = {}


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
        meanValsGauss[singleClass] = (trainClasses[singleClass][randint(1, len(trainClasses[singleClass]), GAUSS_PER_PERSON)])
        covarMatrices[singleClass] = ([np.var(trainClasses[singleClass], axis=0)] * GAUSS_PER_PERSON)
        gaussWeights[singleClass] = np.ones(GAUSS_PER_PERSON) / GAUSS_PER_PERSON;

    print("STEP2 Done: All information for training for all classes are calculated.\n")

    return meanValsGauss, covarMatrices, gaussWeights


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

        for singleClass in trainClasses:
            [gaussWeights[singleClass], meanValsGauss[singleClass], covarMatrices[singleClass], TTL] = train_gmm(trainClasses[singleClass],
                                                                                                                gaussWeights[singleClass],
                                                                                                                meanValsGauss[singleClass],
                                                                                                                covarMatrices[singleClass])

    print("STEP3 Done: Training finished.\n")

    return gaussWeights, meanValsGauss, covarMatrices

# Function loads all evaluation data
#
# @return dictionary containing evaluation data
def loadEvalData():
    print("STEP4: Loading evaluation data.")

    evalData = wav16khz2mfcc(EVAL_DIR)

    print("STEP4 Done: All evaluation data loaded and prepared for classification.\n")

    return evalData

# This function provides the classification of all evaluation data and writes result in file
def classification(evalData, trainClasses, gaussWeights, meanValsGauss, covarMatrices):
    print("STEP5: Classification started.")

    file = open("audio_GMM.txt", "w")

    # For every person to evaluate, calculate the sum of LLs for evaluation data
    for evalPerson in evalData:
        llVals = []
        score = []
        name = evalPerson.split('/')[2]
        name = name.split('.')[0]
        print("Classifing person ", name)
        file.write(name)
        file.write(' ')

        for trainPerson in trainClasses:
            llVals.append(sum(logpdf_gmm(evalData[evalPerson],
                                         gaussWeights[trainPerson],
                                         meanValsGauss[trainPerson],
                                         covarMatrices[trainPerson])))

        file.write(str(llVals[N_OF_CLASSES-1]))

        file.write(' ')

        # Hard decision - if the value of target class is maximal, its 1
        if(np.argmax(llVals) == N_OF_CLASSES-1):
            file.write('1')
        else:
            file.write('0')

        file.write('\n')

    file.close()
    print("STEP5 Done: Classification ended.\n")

    print("Classification is finished. Check file \'audo_GMM.txt\' for the results.")

# main function
def main():
    trainClasses = loadTrainData()

    meanValsGauss, covarMatrices, gaussWeights = getTrainingParams(trainClasses)

    gaussWeights, meanValsGauss, covarMatrices = trainGMMs(trainClasses, gaussWeights, meanValsGauss, covarMatrices)

    evalData = loadEvalData()

    classification(evalData, trainClasses, gaussWeights, meanValsGauss, covarMatrices)


if __name__ == "__main__":
	main()
