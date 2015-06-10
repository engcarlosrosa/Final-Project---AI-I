from __future__ import print_function


import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import csv
import sys
from collections import OrderedDict
import re 
from numpy import genfromtxt
from StringIO import StringIO

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


print(__doc__)
lista1 = [ f for f in listdir("../photos") if isfile(join("../photos",f))]
lista_photos = []
lista_photos2 = []
lista_rgb = []
lista_numero_arthur = []
lista_numero_carlos = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


for item in lista1:
	if item.endswith("jpg") == True or item.endswith("JPG"):
		lista_photos.append(item)
lista_photos.sort(key = lambda x : int(re.findall('\d+', x)[0]))


values = []

for i in range(len(lista_photos)):
	values.append([])
	if lista_photos[i].startswith("arthur") == True:
		lista_numero_arthur.append("1")
	if lista_photos[i].startswith("carlos")==True:
		lista_numero_carlos.append("2")
# as chaves serao os nomes das fotos e seus valores serao os dados RGB de cada pixel
#print lista_numero_arthur
numero =  OrderedDict(zip(lista_photos, values ))


for i in range(0,len(lista_photos)):
	
	img = cv2.imread(lista_photos[i])
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0)
	for (x,y,w,h) in faces:
	    #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    #for (ex,ey,ew,eh) in eyes:
	        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#Aqui redimensionei o tamanho das fotos orginais para o tamanho 37 x 50, baseado nos dados do dataset lfw.

	resized = cv2.resize(img,(37,50), interpolation = cv2.INTER_AREA)    
	lista_photos2.append(resized)
	#print numero.values()[0]
	"""cv2.imshow('img',resized)
	cv2.waitKey(0)
	cv2.destroyAllWindows()"""
# extracao dos valores RGB de cada pixel de cada foto que cortamos anteriormente, que foram armazenadas na lista "Lista_photos2"	
	for w in range(49):     # calculo dos valores de cada pixel da foto 37 x 50
		for j in range(36):
			for k in range(3):
				rgb = resized[w,j,k]
				lista_rgb.append(float(rgb))
	for a in range(len(lista_rgb)):
		lista_rgb[a] = float(lista_rgb[a])/1.0

	
	numero.values()[i].extend(lista_rgb) #o valor de cada chave do dicionario "numero" sao os valores dos pixels dentro 
	del lista_rgb[:]                     # da lista "lista_rgb"
print (numero.keys())
ntitulos = 49  * 36 * 3
lista_titulos = []
lista_titulos.append("CLASS")

for i in range(ntitulos):
	lista_titulos.append("p" + ("%d" % i) )


with open("arthur.csv", 'wb') as csvf:  # aqui comecerei a escrever os dados num arquivo csv
    csvwriter = csv.writer(csvf)  
    csvwriter.writerow(lista_titulos) # criacao da primeira coluna "CLASS" com as classes de cada foto
    for filename, values in numero.items():
    	if filename.startswith("arthur"): 
    		csvwriter.writerow(["1"] + values)
    	elif filename.startswith("carlos"):
    		csvwriter.writerow(["2"] + values)
my_data = genfromtxt("arthur.csv", delimiter = ",")  # geracaoo de uma matriz dos dados RGB

dataset = my_data[1:,1:].astype(float)
#print (dataset)


Y = my_data[:,0] # classes
Z = my_data[0] # numero de cada valor(p1,p2,p3 ...)
#rint (Y)
#print (Z)
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
"""A partir daqui comeca o treinamento dos dados para posterior reconhecimento das pessoas
baseada no dataset do lfw
"""
# introspect the images arrays to find the shapes (for plotting)
 
n_samples = 40
h = 50
w = 37

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = dataset
print(X[0])
n_features = X.shape[1]

# the label to predict is the id of the person
y = Y[1:]
print (y)
target_names = ["arthur","carlos"]
n_classes = 2

print("Total dataset size: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

