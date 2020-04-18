from ikrlib import png2fea
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

WIDTH = 80
HEIGHT = 80
CLASSES = []

# Displaying Eigenfaces
def show_eigenfaces(pca):
	fig, axes = plt.subplots(8, 8, figsize=(9, 4),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(80, 80), cmap='gray')
	plt.show()

# # # Getting all training data from data/train
print("STEP1: getting training and evaluation data")
toTrain = []
filenames= os.listdir("data/train")

for filename in filenames: # loop through all the files and folders
    person = png2fea("data/train/" + filename).values()
    CLASSES.extend([filename] * len(person))  # adding given folder name as one class(one person)
    toTrain.extend(person)                    # adding feature of one person in the arre

# # # Getting all evaluation data from data/evaluate
toEvaluate = []
toEvalNames = []

evaluateDict = png2fea('data/eval')

toEvalNames.extend(evaluateDict.keys())
temp = evaluateDict.values()
toEvaluate.extend(temp)
print("STEP1 done.")

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
# value of components according to the graph that can be shown
components = 64

# PCA performing -> extracting eigens
print("STEP3: performimg pca")
pca = PCA(n_components = components, svd_solver = 'full', whiten = True, random_state = 10).fit(toTrain_2d)
print("STEP3 done.")

# To see the graph of PCA and all eigenfaces, set this variable to 1
showPCAandEigens = 0
if(showPCAandEigens == 1):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()

    show_eigenfaces(pca)

eigenfaces = pca.components_.reshape((components, WIDTH, HEIGHT))

print("STEP4: Input data projection to eigens orthogonal basis")
trainPCA = pca.transform(toTrain_2d)
testPCA = pca.transform(toEvaluate_2d)
print("STEP4 done.")


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

print("STEP6: fitting the classifier")
classifier = classifier.fit(trainPCA, CLASSES)
print("STEP6 done.")


print("STEP7: Predicting people's names on the test set")

probaClass = classifier.predict_proba(testPCA)

#print(classification_report(toEvalNames, probaClass))

file = open("xjezik03_xkubik34_image_PCA_MLPerceptron.txt", "w")
for name, proba in zip(toEvalNames, probaClass):
    name = name.split('/')[2]
    name = name.split('.')[0]

    if(proba.tolist()[19] >= 0.5):
        HD = 1
    else:
        HD = 0

    file.write("{0} {1} {2}\n".format(name, proba.tolist()[19], HD))

file.close()
