#Decision Tree model with scaler object
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load preprocessed data
x_train = np.load("dataset/x_train.npy")
x_test = np.load("dataset/x_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

model_DT = DecisionTreeClassifier() #initialising model
model_DT.fit(x_train, y_train) #training model

predict = model_DT.predict(x_test) #testing
for i in range(len(predict)): #loop for results
    print(y_test[i], predict[i])

accuracy = accuracy_score(y_test, predict) #Accuracy
cm = confusion_matrix(y_test, predict) #Confusion Matrix
report = classification_report(y_test, predict, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) #Classification report
print("\nDT Accuracy :", accuracy*100)
print("\nDT Confusion Matrix : \n", cm)
print("\nDT Classification Report : \n", report)

model_save = input("Do you want to save this SVC model ? (Y/N)")
match model_save:
    case ("Y" | "y"):
        with open("models/model_DT.pkl", "wb") as f: #saving the DT model as model_DT.pkl
            pickle.dump(model_DT, f)
        print ("Decision Tree Classifier model saved as 'model_DT.pkl'")
        print("Model saved as 'model_DT.pkl")
    case ("N" | "n"):
        print("Very well")

