#importing stuff
import cv2
from cv2 import VideoCapture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import os, ssl, time
from PIL import Image
import PIL.ImageOps

#Fetching the data
X = np.load('image.npz')['arr_0']
Y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(Y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

#Splitting the data and scaling it
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=9, train_size=3500, test_size=500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
