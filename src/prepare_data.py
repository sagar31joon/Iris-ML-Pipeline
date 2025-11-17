import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

#iris_dataset = load_dataset("sasageooo/iris-flower-classification") #loading dataset from hugging face

#iris_dataset['train'].to_csv("iris_dataset.csv") #saving dataset as CSV file

df = pd.read_csv("dataset/iris_raw.csv")
#print(df)
print("\nRaw dataset : ")
print(df.head())
print (df.shape) #rows x column numbers
#print(df.info()) 
#print(df.describe()) 

#label encoding
le = LabelEncoder()
df['species'] = le.fit_transform(df['species']) #encoding species column
print("\nEncoded dataset : ")
print(df.head())
print (df.shape) #rows x column numbers

df.to_csv("dataset/encoded.csv", index=False) #creating processed dataset file for model

data_value = df.values #slicing dataset
x = data_value[:, 0:4]
y = data_value[:, 4]

Train_Split = 0.8 #variable
Random = 42 #variable

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = Train_Split, random_state = Random) #splitting

scaler = StandardScaler() #initialising scaler
x_train_scaled = scaler.fit_transform(x_train) #standardise and transform data
x_test_scaled = scaler.transform(x_test) #only transform data


scaler_save = input("Do you want to save this scaler model ? (Y/N)")
match scaler_save:
    case ("Y" | "y"):
        with open("models/scaler.pkl", "wb") as f: #saving scaler model for reuse
            pickle.dump(scaler, f)
        print("Scaler model saved as 'scaler.pkl'")
    case ("N" | "n"):
        print("Very well")

scaled_array_save = input("Do you want to save these scaled arrays ? (Y/N)") # saving the arrays for models
match scaled_array_save:
    case ("Y" | "y"):
        np.save("dataset/x_train.npy", x_train_scaled)
        np.save("dataset/x_test.npy", x_test_scaled)
        np.save("dataset/y_train.npy", y_train)
        np.save("dataset/y_test.npy", y_test)
        
        print("\nPreprocessing complete!")
        print("Saved:")
        print(" - x_train.npy")
        print(" - x_test.npy")  
        print(" - y_train.npy")
        print(" - y_test.npy")
    case ("N" | "n"): # exiting save
        print("Very well")
