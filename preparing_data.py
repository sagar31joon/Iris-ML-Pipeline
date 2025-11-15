import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split

#iris_dataset = load_dataset("sasageooo/iris-flower-classification") #loading dataset from hugging face

#iris_dataset['train'].to_csv("iris_dataset.csv") #saving dataset as CSV file

df = pd.read_csv("dataset/iris_raw.csv")
print(df)
#print(df.head()) 
print(df.info()) 
print(df.describe()) 

#plot1 = sns.pairplot(df, hue = "species") #plotting dataset
#print(plot1)
#plt.show()

from sklearn.preprocessing import LabelEncoder #label encoding
le = LabelEncoder()
df['species'] = le.fit_transform(df['species']) #encoding species column

dataset_values = df.values #slicing dataset
x = dataset_values[:, 0:4]
y = dataset_values[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) #splitting into train & test sets
print("Train split : ")
print(x_train, y_train)
print("Test split : ")
print(x_test, y_test)

from sklearn.preprocessing import StandardScaler #standardizing and transforming data so that all features have mean of 0 and sd of 1
sc = StandardScaler() 
x_train = scaler.fit_transform(x_train) #fit and transform train data
x_test = scaler.transform(x_test) #only transform test data

print("Train split : ")
print(x_train, y_train)
print("Test split : ")
print(x_test, y_test)



