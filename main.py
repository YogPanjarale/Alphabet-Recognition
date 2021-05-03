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