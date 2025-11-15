#Logistic Refression trainer and sclaler model initialiser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset/iris_processed.csv") #loading processed dataset file
print(df.head())

data_value = df.values #slicing dataset
x = data_value[:, 0:4]
y = data_value[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #test and train split
print("Train split X shape:", x_train.shape)
print("Test split X shape :", x_test.shape)

scaler = StandardScaler() #scaling
x_train = scaler.fit_transform(x_train) #standardise and transform data
x_test = scaler.transform(x_test) # only transform data

print("\nScaled TRAIN:")
print(x_train)

print("\nScaled TEST:")
print(x_test)

with open("models/scaler.pkl", "wb") as f: #saving scaler model for reuse
    pickle.dump(scaler, f)








