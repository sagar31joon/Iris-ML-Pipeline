# SVM model with scaler object
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVC



df = pd.read_csv("dataset/iris_processed.csv") #loading processed dataset

data_value = df.values #slicing dataset
x = data_value[:, 0:4]
y = data_value[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #train and test split

with open("models/scaler.pkl", "rb") as f: #loading scaler model
    scaler = pickle.load(f)



