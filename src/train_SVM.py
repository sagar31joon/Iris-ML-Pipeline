# Suport Vector Model with scaler object
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load preprocessed data
x_train = np.load("dataset/x_train.npy")
x_test = np.load("dataset/x_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

model_svc = SVC() #initialising SVC model
model_svc.fit(x_train, y_train) #training model

predict = model_svc.predict(x_test) # testing
for i in range(len(predict)): #loop for checking results
    print(y_test[i], predict[i])

accuracy = accuracy_score(y_test, predict) #Accuracy
cm = confusion_matrix(y_test, predict) #Confusion Matrix
report = classification_report(y_test, predict, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) #Classification report
print("\nSVC Accuracy :", accuracy*100)
print("\nSVC Confusion Matrix : \n", cm)
print("\nSVC Classification Report : \n", report)

model_save = input("Do you want to save this SVC model ? (Y/N)")
match model_save:
    case ("Y" | "y"):
        with open("models/model_SVC.pkl", "wb") as f: #saving the model as model_SVC.pkl
            pickle.dump(model_svc, f)
        print("Support Vector Classifier saved as 'model_SVC.pkl'")
        print("Model saved as 'model_SVM.pkl")
    case ("N" | "n"):
        print("Very well")











