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
#from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# coloca os nomes dos documentos do arquivo "photos" numa lista.
print(__doc__)
lista1 = [ f for f in listdir("../photos") if isfile(join("../photos",f))]
lista_photos = []
lista_photos2 = []
lista_rgb = []
lista_numero_arthur = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# seleciona todas as fotos(documento jpg) do arquivo.
for item in lista1:
	if item.endswith("jpg") == True:
		lista_photos.append(item)
lista_photos.sort(key = lambda x : int(re.findall('\d+', x)[0]))


values = []
# seleciona as fotos "arthur"
for i in range(len(lista_photos)):
	values.append([])
	if lista_photos[i].startswith("arthur") == True:
		lista_numero_arthur.append("1")

# criacao de um dicionario
numero =  OrderedDict(zip(lista_photos, values ))


for i in range(0,len(lista_photos)-15):
	
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
	
	resized = cv2.resize(img,(62,74), interpolation = cv2.INTER_AREA)    
	lista_photos2.append(resized)
	#print numero.values()[0]
	cv2.imshow('img',resized)
	cv2.waitKey(0)
	cv2.destroyAllWindows()






















