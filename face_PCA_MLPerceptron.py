from ikrlib import png2fea
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

COMPONENTS = 64             # value of components according to the graph that can be shown
WIDTH = 80
HEIGHT = 80

CLASSES = []

TRAIN_DIR = "data/train"
EVAL_DIR = "data/eval2"

# Function that displays eigenfaces created form imput images
#
# @param pca Principal component analysis result
#
# @post image of 64 eigenfaces is plotted
def showEigenfaces(pca):
	fig, axes = plt.subplots(8, 8, figsize=(9, 4),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(80, 80), cmap='gray')
	plt.show()

# Function that performs several needed actions with arrays representing input images
#
# @pre toTrain and toEvaluate are not empty
#
# @param toTrain array representing all training pictures
#
# @param toEvaluate array representing all evaluated pictures
#
# @return arrays will be transfromed into NumPy arrays and its dim will be reduced into 2d
def imagesModify(toTrain, toEvaluate):
    # # # Transforming train and evaluate data in Numpy arrays
    print("STEP2: various arrays of images actions")

    toTrain = np.array(toTrain)
    toEvaluate = np.array(toEvaluate)

    # # # Transforming the dimensionality of training set into 2d
    nsamples, nx, ny = toTrain.shape
    toTrain_2d = toTrain.reshape((nsamples, nx * ny))

    # # # Transforming the dimensionality of evaluation set into 2d
    nsamples, nx, ny = toEvaluate.shape
    toEvaluate_2d = toEvaluate.reshape((nsamples, nx * ny))

    print("STEP2 done.")

    return toTrain_2d, toEvaluate_2d

# Function creates a PCA and fits our training set into it
#
# @param toTrain_2d Training picture set
#
# @return modified for better classification
def performPCA(toTrain_2d):
    # PCA performing -> extracting eigens
    print("STEP3: performimg pca")
    pca = PCA(n_components = COMPONENTS, svd_solver = 'full', whiten = True, random_state = 10).fit(toTrain_2d)
    print("STEP3 done.")

    return pca

# Auxilliary function to show the PCA value so that you know what number of components to choose and also all eigenfaces
def showPCAandEigens(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()

    showEigenfaces(pca)

# Function to project eigenfaces into ots orthogonal bases
#
# @param toTrain_2d Training pictures set
#
# @param toEvaluate_2d Evaluating pictures set
#
# @param pca
#
# @return train and evaluate set transformed
def projectEigens(toTrain_2d, toEvaluate_2d, pca):
    print("STEP4: Input data projection to eigens orthogonal bases")
    trainPCA = pca.transform(toTrain_2d)
    evalPCA = pca.transform(toEvaluate_2d)
    print("STEP4 done.")

    return trainPCA, evalPCA

# Function to set up a classifier
#
# @return classifier to classify out sets
def setUpClassifier():
    print("STEP5: setting up the classifier")
    classifier = MLPClassifier( hidden_layer_sizes=(800,),
                                solver = 'adam',
                                activation='logistic',
                                batch_size = 'auto',
                                alpha = 0.0001,
                                max_iter = 500,
                                verbose = True,
                                shuffle = True,
                                random_state = 1)
    print("STEP5 done.")

    return classifier

# Function fits our training set into a classifier
def fitClassifier(classifier, trainPCA, CLASSES):
    print("STEP6: fitting the classifier")
    classifier = classifier.fit(trainPCA, CLASSES)
    print("STEP6 done.")

    return classifier

# Function predits classes of all pics from eval set and writes it in a file
#
# Form of output:
# eval_picture probability_of_being_target hard_decision
# where hard_decision == 1 if probability_of_being_target >= 0.5, otherwise 0
def predict(classifier, evalPCA, toEvalNames):
    print("STEP7: Predicting people's names on the test set")

    probaClass = classifier.predict_proba(evalPCA)

    #print(classification_report(toEvalNames, probaClass))

    file = open("xjezik03_xkubik34_image_PCA_MLPerceptron.txt", "w")
    for name, proba in zip(toEvalNames, probaClass):
        name = name.split('/')[2]
        name = name.split('.')[0]

        if(proba.tolist()[19] >= 0.5):
            HD = 1
        else:
            HD = 0

        file.write(name)
        file.write(" ")
        file.write(str(proba.tolist()[19])) # target is 20th class.
        file.write(" ")
        file.write(str(HD))
        file.write("\n")

    file.close()

    print("STEP7 done.\n")



def main():
    # # # Getting all training data from data/train
    print("STEP1: getting training and evaluation data")
    toTrain = []
    filenames= os.listdir(TRAIN_DIR)

    for filename in filenames:
        person = png2fea(TRAIN_DIR + "/" + filename).values()
        CLASSES.extend([filename] * len(person))  # adding given folder name as one class(one person)
        toTrain.extend(person)                    # adding feature of one person in the arre

    # # # Getting all evaluation data from data/evaluate
    toEvaluate = []
    toEvalNames = []

    evaluateDict = png2fea(EVAL_DIR)

    toEvalNames.extend(evaluateDict.keys())
    temp = evaluateDict.values()
    toEvaluate.extend(temp)
    print("STEP1 done.")

    toTrain_2d, toEvaluate_2d = imagesModify(toTrain, toEvaluate)

    pca = performPCA(toTrain_2d)

    # To see the graph of PCA and all eigenfaces, set this variable to 1
    showPCAandEigens = 0
    if(showPCAandEigens == 1):
        showPCAandEigens(pca)

    eigenfaces = pca.components_.reshape((COMPONENTS, WIDTH, HEIGHT))

    trainPCA, evalPCA = projectEigens(toTrain_2d, toEvaluate_2d, pca)

    classifier = setUpClassifier()

    classifier = fitClassifier(classifier, trainPCA, CLASSES)

    predict(classifier, evalPCA, toEvalNames)

    print("CLASSIFICATION is done successfuly. \nCheck file xjezik03_xkubik34_image_PCA_MLPerceptron.txt for the results.")

    sys.exit(0)

if __name__ == "__main__":
	main()
